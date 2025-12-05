pkgs_root <- c("brms", "cmdstanr", "jsonlite", "data.table")

missing_root <- pkgs_root[!vapply(pkgs_root, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_root) > 0) {
    stop("Missing required packages: ", paste(missing_root, collapse = ", "))
}

# full recursive deps (Depends, Imports, LinkingTo, Suggests)
deps_list <- tools::package_dependencies(
    pkgs_root,
    recursive = TRUE,
    which = c("Depends", "Imports", "LinkingTo")
)

all_deps <- unique(c(pkgs_root, unlist(deps_list, use.names = FALSE)))

ip_all <- installed.packages()
present <- intersect(all_deps, rownames(ip_all))

missing <- setdiff(all_deps, present)
if (length(missing) > 0) {
    stop("Dependencies not installed: ", paste(missing, collapse = ", "))
}

ip <- as.data.frame(
    ip_all[present, c("Package", "Version", "LibPath", "Priority")],
    stringsAsFactors = FALSE
)

ip$Priority[is.na(ip$Priority)] <- ""

# Drop base/recommended packages – assume they exist on target system
ip <- ip[is.na(ip$Priority) |
            ip$Priority == ""  |
            !(ip$Priority %in% c("base", "recommended")),
            , drop = FALSE]

# Ensure cmdstanr is loaded and cmdstan is configured
library(cmdstanr)
cs_path <- cmdstanr::cmdstan_path()
if (is.na(cs_path) || !nzchar(cs_path)) {
    stop("cmdstan_path() is empty – cmdstan is not installed/configured.")
}
cs_ver <- as.character(cmdstanr::cmdstan_version())

ver <- R.Version()
r_ver_str <- paste(ver$major, ver$minor, sep = ".")

info <- list(
    r_version       = r_ver_str,
    cmdstan_path    = cs_path,
    cmdstan_version = cs_ver,
    packages        = ip
)

# Return as JSON for Python
jsonlite::toJSON(info, dataframe = "rows", auto_unbox = TRUE)