# third_party

本目录只存放通过 git submodule 固定版本的第三方源码，用于离线构建和可复现构建。不要把 tarball 或复制粘贴的源码直接提交到这里。

| 路径 | 版本 | 固定 commit | 许可证 | 用途 |
|---|---|---|---:|---|
| `third_party/googletest` | v1.18.0 | `063de7e9578f82b369302001269680b4b1553359` | BSD-3-Clause | C++ 单元测试（`gtest` / `gtest_main`） |
| `third_party/tokenizers-cpp` | main (`c586c52`) | `c586c52f93f7b060753bd2388eb96a105cb7374d` | Apache-2.0 | HuggingFace `tokenizer.json` 的 C++ 绑定；HF-only 构建只需要 `huggingface_tokenizer.cc` + Rust `tokenizers` |

## 初始化

```bash
# tokenizers-cpp 的 msgpack/sentencepiece 子模块不参与 HF-only 构建，
# 不要加 --recursive，否则会去拉不可达的旧 commit。
git submodule update --init third_party/googletest third_party/tokenizers-cpp
```

`tokenizers-cpp` 的 Rust wrapper 需要 `cargo`；默认从本地 cargo cache 离线构建，可传
`-DLLMCP_CARGO_OFFLINE=OFF` 允许联网。

如果是第一次 clone 仓库：

```bash
git clone <repo-url>
cd llmcp
git submodule update --init third_party/googletest third_party/tokenizers-cpp
```

## 版本升级流程

不要直接在子模块里 `git pull` 后提交。按下面步骤升级，保证根仓库记录的是明确的 commit：

```bash
cd third_party/googletest
git fetch --tags
git checkout vX.Y.Z
cd ../..
git add third_party/googletest .gitmodules
git commit -m "Bump googletest to vX.Y.Z"
```

`googletest` 同理。升级后同步更新本文件的版本表和 `docs/design.md` 的依赖清单。

## CMake 接入

根 `CMakeLists.txt` 只需要：

```cmake
add_subdirectory(third_party)
```

`third_party/CMakeLists.txt` 会关闭 vendored 项目的安装和测试目标，避免污染主工程：

- `googletest`：提供 `GTest::gtest`、`GTest::gtest_main`

使用示例：

```cmake
target_link_libraries(llmcp_unittests PRIVATE GTest::gtest_main)
```

## 注意

- 当前 submodule 是浅克隆（`--depth 1`），日常构建不需要完整历史。
- 如果需要查看旧版本或执行 `git submodule update --remote`，先在子模块内执行 `git fetch --unshallow`。
- 不要把 `third_party/` 下的构建产物提交进仓库；它们已在 `.gitignore` 中排除。
