# brmspy/rsession.py
"""
Generic restartable R session.

R runs in a subprocess. Uses your existing py_to_r/r_to_py converters.
Large arrays transfer via shared memory automatically.
"""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing import shared_memory
from multiprocessing.connection import Connection
import numpy as np
import pickle
import os
import atexit
import threading
import re
from typing import Optional, List, Any, Tuple, Dict
from dataclasses import dataclass, field


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class RSessionConfig:
    """Configuration for RSession."""
    
    # Shared memory size (default 500MB)
    shm_size: int = 500 * 1024 * 1024
    
    # Arrays larger than this go through shared memory
    shm_threshold: int = 1 * 1024 * 1024  # 1MB
    
    # R packages to load on startup
    packages: List[str] = field(default_factory=list)
    
    # R code to run on startup
    init_code: str = ""
    
    # Timeout for commands (None = no timeout)
    timeout: Optional[float] = None
    
    # Suppress R startup messages
    quiet: bool = True


# =============================================================================
# SHARED MEMORY HELPERS
# =============================================================================

def _serialize_with_shm(obj: Any, shm_buf, threshold: int) -> Tuple[bytes, Dict[str, dict]]:
    """
    Serialize object, putting large arrays in shared memory.
    
    Returns (pickled_data, {array_id: {'offset': int, 'shape': tuple, 'dtype': str}})
    """
    shm_arrays = {}
    offset = 0
    
    def replace_large_arrays(o, path="root"):
        nonlocal offset
        
        if isinstance(o, np.ndarray) and o.nbytes >= threshold:
            # Write to shared memory
            arr = np.ascontiguousarray(o)
            nbytes = arr.nbytes
            
            # Align to 64 bytes
            aligned_offset = (offset + 63) & ~63
            
            if aligned_offset + nbytes > len(shm_buf):
                raise MemoryError(
                    f"Shared memory exhausted. Need {aligned_offset + nbytes}, "
                    f"have {len(shm_buf)}. Increase shm_size."
                )
            
            shm_buf[aligned_offset:aligned_offset + nbytes] = arr.tobytes()
            
            array_id = f"__shm_{len(shm_arrays)}__"
            shm_arrays[array_id] = {
                'offset': aligned_offset,
                'shape': arr.shape,
                'dtype': str(arr.dtype),
                'nbytes': nbytes
            }
            
            offset = aligned_offset + nbytes
            return {'__shm_ref__': array_id}
        
        elif isinstance(o, dict):
            return {k: replace_large_arrays(v, f"{path}.{k}") for k, v in o.items()}
        
        elif isinstance(o, (list, tuple)):
            result = [replace_large_arrays(v, f"{path}[{i}]") for i, v in enumerate(o)]
            return type(o)(result)
        
        else:
            return o
    
    replaced = replace_large_arrays(obj)
    pickled = pickle.dumps(replaced)
    
    return pickled, shm_arrays


def _deserialize_with_shm(pickled: bytes, shm_arrays: dict, shm_buf) -> Any:
    """Deserialize object, reading large arrays from shared memory."""
    
    obj = pickle.loads(pickled)
    
    def restore_arrays(o):
        if isinstance(o, dict):
            if '__shm_ref__' in o:
                array_id = o['__shm_ref__']
                info = shm_arrays[array_id]
                arr = np.ndarray(
                    shape=info['shape'],
                    dtype=np.dtype(info['dtype']),
                    buffer=shm_buf,
                    offset=info['offset']
                ).copy()
                return arr
            else:
                return {k: restore_arrays(v) for k, v in o.items()}
        
        elif isinstance(o, (list, tuple)):
            return type(o)(restore_arrays(v) for v in o)
        
        else:
            return o
    
    return restore_arrays(obj)


# =============================================================================
# WORKER PROCESS
# =============================================================================

def _worker_main(
    cmd_pipe: Connection,
    result_pipe: Connection,
    shm_name: str,
    shm_size: int,
    config_pickle: bytes,
    converter_module: str
):
    """Worker process - owns the embedded R interpreter."""
    import signal
    signal.signal(signal.SIGTERM, lambda *_: os._exit(0))
    
    config: RSessionConfig = pickle.loads(config_pickle)
    
    # Import converters
    try:
        import importlib
        conv_mod = importlib.import_module(converter_module)
        py_to_r = conv_mod.py_to_r
        r_to_py = conv_mod.r_to_py
    except Exception as e:
        result_pipe.send({'status': 'error', 'error': f"Failed to import converters: {e}"})
        os._exit(1)
    
    # Import rpy2
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import globalenv
    except ImportError as e:
        result_pipe.send({'status': 'error', 'error': f"Failed to import rpy2: {e}"})
        os._exit(1)
    
    # Attach shared memory
    shm = shared_memory.SharedMemory(name=shm_name)
    
    # Initialize R
    try:
        if config.quiet:
            ro.r('options(warn = -1)')
        
        for pkg in config.packages:
            if config.quiet:
                ro.r(f'suppressPackageStartupMessages(library({pkg}))')
            else:
                ro.r(f'library({pkg})')
        
        if config.init_code:
            ro.r(config.init_code)
            
    except Exception as e:
        result_pipe.send({'status': 'error', 'error': f"R init failed: {e}"})
        os._exit(1)
    
    result_pipe.send({'status': 'ready'})
    
    # Main loop
    while True:
        try:
            cmd = cmd_pipe.recv()
        except (EOFError, BrokenPipeError):
            break
        
        if cmd['type'] == 'SHUTDOWN':
            break
        
        try:
            result = _execute(cmd, shm, config, py_to_r, r_to_py, ro)
            result_pipe.send({'status': 'ok', **result})
        except Exception as e:
            import traceback
            result_pipe.send({
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            })
    
    shm.close()
    os._exit(0)


def _execute(cmd: dict, shm, config, py_to_r, r_to_py, ro) -> dict:
    """Execute a command."""
    
    code = cmd['code']
    args_pickle = cmd['args_pickle']
    args_shm = cmd['args_shm']
    kwargs_pickle = cmd['kwargs_pickle']
    kwargs_shm = cmd['kwargs_shm']
    
    # Deserialize arguments
    args = _deserialize_with_shm(args_pickle, args_shm, shm.buf)
    kwargs = _deserialize_with_shm(kwargs_pickle, kwargs_shm, shm.buf)
    
    # Convert to R
    r_args = [py_to_r(a) for a in args]
    r_kwargs = {k: py_to_r(v) for k, v in kwargs.items()}
    
    # Detect if code is a function definition
    code_stripped = code.strip()
    is_function = (
        code_stripped.startswith('function') or 
        code_stripped.startswith('\\(') or  # R 4.1+ shorthand
        re.match(r'^[a-zA-Z_][a-zA-Z0-9_.]*\s*\(', code_stripped)  # function call like foo(...)
    )
    
    if is_function and (args or kwargs):
        # It's a function - call it with args/kwargs
        r_func = ro.r(code)
        result = r_func(*r_args, **r_kwargs)
    else:
        # It's code - inject kwargs as variables, execute
        for name, r_obj in r_kwargs.items():
            ro.globalenv[name] = r_obj
        
        # If there are positional args, assign them as .arg1, .arg2, etc.
        for i, r_obj in enumerate(r_args):
            ro.globalenv[f'.arg{i + 1}'] = r_obj
        
        result = ro.r(code)
    
    # Convert result to Python
    py_result = r_to_py(result)
    
    # Serialize result with shared memory for large arrays
    result_pickle, result_shm = _serialize_with_shm(py_result, shm.buf, config.shm_threshold)
    
    return {
        'result_pickle': result_pickle,
        'result_shm': result_shm
    }


# =============================================================================
# MAIN SESSION CLASS
# =============================================================================

class RSession:
    """
    A restartable R session.
    
    R runs in a subprocess with your existing py_to_r/r_to_py converters.
    Large arrays transfer via shared memory automatically.
    
    Example
    -------
    >>> session = RSession(packages=['dplyr'])
    >>> 
    >>> # Call R function with args
    >>> session.run('function(a, b) a + b', 1, 2)
    3
    >>> 
    >>> # Run code with injected variables  
    >>> session.run('head(df, n)', df=my_df, n=5)
    >>> 
    >>> # Restart R completely
    >>> session.restart()
    """
    
    def __init__(
        self,
        converter_module: str = "brmspy.converters",
        packages: Optional[List[str]] = None,
        init_code: str = "",
        shm_size: int = 500 * 1024 * 1024,
        shm_threshold: int = 1 * 1024 * 1024,
        timeout: Optional[float] = None,
        quiet: bool = True
    ):
        """
        Initialize R session.
        
        Parameters
        ----------
        converter_module : str
            Module path containing py_to_r and r_to_py functions
        packages : list of str
            R packages to load on startup
        init_code : str
            R code to run on startup
        shm_size : int
            Shared memory size in bytes (default 500MB)
        shm_threshold : int
            Arrays larger than this use shared memory (default 1MB)
        timeout : float, optional
            Command timeout in seconds
        quiet : bool
            Suppress R startup messages
        """
        self._converter_module = converter_module
        self._config = RSessionConfig(
            shm_size=shm_size,
            shm_threshold=shm_threshold,
            packages=packages or [],
            init_code=init_code,
            timeout=timeout,
            quiet=quiet
        )
        
        self._shm: Optional[shared_memory.SharedMemory] = None
        self._worker: Optional[mp.Process] = None
        self._cmd_pipe: Optional[Connection] = None
        self._result_pipe: Optional[Connection] = None
        self._running = False
        self._lock = threading.RLock()
        
        self._start()
        atexit.register(self.shutdown)
    
    def _start(self):
        """Start the worker process."""
        with self._lock:
            self._shm = shared_memory.SharedMemory(
                create=True,
                size=self._config.shm_size
            )
            
            parent_cmd, child_cmd = mp.Pipe()
            child_result, parent_result = mp.Pipe()
            
            self._cmd_pipe = parent_cmd
            self._result_pipe = parent_result
            
            self._worker = mp.Process(
                target=_worker_main,
                args=(
                    child_cmd,
                    child_result,
                    self._shm.name,
                    self._shm.size,
                    pickle.dumps(self._config),
                    self._converter_module
                ),
                daemon=True
            )
            self._worker.start()
            
            # Wait for ready
            response = self._result_pipe.recv()
            if response['status'] != 'ready':
                self._worker.terminate()
                raise RuntimeError(f"R worker failed: {response.get('error', 'Unknown error')}")
            
            self._running = True
    
    def _stop(self):
        """Stop the worker process."""
        if self._cmd_pipe is None or self._worker is None:
            return
        with self._lock:
            if self._running:
                try:
                    self._cmd_pipe.send({'type': 'SHUTDOWN'})
                    self._worker.join(timeout=5)
                except:
                    pass
                
                if self._worker.is_alive():
                    self._worker.terminate()
                    self._worker.join(timeout=2)
                    if self._worker.is_alive():
                        self._worker.kill()
                
                self._running = False
            
            if self._shm:
                try:
                    self._shm.close()
                    self._shm.unlink()
                except:
                    pass
                self._shm = None
    
    @property
    def is_running(self) -> bool:
        """Check if session is running."""
        return self._running and (self._worker is not None and self._worker.is_alive())
    
    def restart(self):
        """Restart R session completely. All R state is cleared."""
        with self._lock:
            self._stop()
            self._start()
    
    def shutdown(self):
        """Shut down the R session."""
        self._stop()
    
    def run(self, code: str, *args, **kwargs) -> Any:
        """
        Run R code or call R function.
        
        Parameters
        ----------
        code : str
            R code to execute. Can be:
            - A function: 'function(a, b) a + b'
            - A function call: 'mean(x)'  
            - Any R code: 'x + y'
        *args
            Positional arguments (passed to function, or available as .arg1, .arg2, ...)
        **kwargs
            Named arguments (passed to function, or injected as R variables)
        
        Returns
        -------
        Any
            Result converted to Python via r_to_py
        
        Examples
        --------
        >>> # Call anonymous function
        >>> session.run('function(a, b) a + b', 1, 2)
        3
        
        >>> # Call with named args
        >>> session.run('function(x, n = 5) head(x, n)', my_data, n=10)
        
        >>> # Run code with variables
        >>> session.run('lm(y ~ x, data = df)', df=my_dataframe)
        
        >>> # Access positional args as .arg1, .arg2, ...
        >>> session.run('.arg1 + .arg2', 10, 20)
        30
        """
        if not self.is_running:
            raise RuntimeError("R session not running. Call restart().")
        
        assert self._cmd_pipe
        assert self._result_pipe
        assert self._shm
        
        with self._lock:
            # Serialize args with shared memory for large arrays
            args_pickle, args_shm = _serialize_with_shm(
                list(args), 
                self._shm.buf, 
                self._config.shm_threshold
            )
            kwargs_pickle, kwargs_shm = _serialize_with_shm(
                kwargs,
                self._shm.buf,
                self._config.shm_threshold
            )
            
            cmd = {
                'type': 'RUN',
                'code': code,
                'args_pickle': args_pickle,
                'args_shm': args_shm,
                'kwargs_pickle': kwargs_pickle,
                'kwargs_shm': kwargs_shm
            }
            
            try:
                self._cmd_pipe.send(cmd)
                
                if self._config.timeout:
                    if not self._result_pipe.poll(self._config.timeout):
                        raise TimeoutError(f"R command timed out after {self._config.timeout}s")
                
                response = self._result_pipe.recv()
                
            except (BrokenPipeError, EOFError) as e:
                self._running = False
                raise RuntimeError(f"R worker died: {e}")
            
            if response['status'] == 'error':
                raise RSessionError(
                    response['error'],
                    response.get('traceback', '')
                )
            
            # Deserialize result
            return _deserialize_with_shm(
                response['result_pickle'],
                response['result_shm'],
                self._shm.buf
            )
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.shutdown()
    
    def __del__(self):
        try:
            self.shutdown()
        except:
            pass


class RSessionError(Exception):
    """Exception from R code execution."""
    
    def __init__(self, message: str, traceback: str = ""):
        self.r_traceback = traceback
        super().__init__(f"{message}\n\nR traceback:\n{traceback}" if traceback else message)


# =============================================================================
# MODULE-LEVEL CONVENIENCE
# =============================================================================

_default_session: Optional[RSession] = None


def get_session(**kwargs) -> RSession:
    """Get or create the default R session."""
    global _default_session
    if _default_session is None or not _default_session.is_running:
        _default_session = RSession(**kwargs)
    return _default_session


def run(code: str, *args, **kwargs) -> Any:
    """Run R code using the default session."""
    return get_session().run(code, *args, **kwargs)


def restart():
    """Restart the default R session."""
    global _default_session
    if _default_session is not None:
        _default_session.restart()