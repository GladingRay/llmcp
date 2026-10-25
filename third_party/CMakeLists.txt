# Vendored third-party dependencies. Versions are pinned by git submodules.
#
# Keep this file free of project-specific logic: it only exposes the libraries
# that the rest of the build links against.

# utf8proc: Unicode normalization and character classification for the tokenizer.
set(UTF8PROC_INSTALL OFF CACHE BOOL "" FORCE)
set(UTF8PROC_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
add_subdirectory(utf8proc EXCLUDE_FROM_ALL)

# GoogleTest: C++ unit tests. BUILD_GMOCK can be enabled later if needed.
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
add_subdirectory(googletest EXCLUDE_FROM_ALL)
