# Vendored third-party dependencies. Versions are pinned by git submodules.
#
# Keep this file free of project-specific logic: it only exposes the libraries
# that the rest of the build links against.

# GoogleTest: C++ unit tests. BUILD_GMOCK can be enabled later if needed.
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
add_subdirectory(googletest EXCLUDE_FROM_ALL)
