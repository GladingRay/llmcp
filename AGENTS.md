# llmcp
目标：基于MLIR，实现一个以CPU为后端运行的 qwen3.5 4B 大模型的编译器。
## instructions
1. 在实现过程中如果功能点或者bugfix完成可以自动commit。
2. commit message与代码中的注释需要使用英文。
3. 文档使用中文书写。
4. 代码风格、命名规范与注释格式与MLIR保持一致。
5. 项目主体使用C++实现，可辅以部分python与shell脚本。
6. 不同的模块要放在不同的文件中，不同的大组件要放到不同的目录中。
7. 我本机上装的是homebrew的llvm@20。
8. qwen3.5 4B 的路径在 `/Users/guorui/Work/models/qwen3.5-4b` 。
9. python环境：`/Users/guorui/Work/models/venv314`。