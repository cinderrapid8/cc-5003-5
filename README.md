# band-align

远程乐队排练录音对齐工具。导入各成员的分轨 WAV，以共享的拍手/节拍为参考，
估计每条成员轨相对参考轨的**固定延迟**与**采样时钟漂移**，用一条连续的分段
线性时间映射做重采样，输出仍然可以分轨编辑的对齐工程。

名义采样率一致并不代表两台录音机的实际时钟相同：本工具不做“一次平移”，而是
从分布在整条录音上的可信锚点拟合 `project_time = slope * member_time + offset`，
其中 `slope = 1 + drift`。

## 工作区结构

```
Cargo.toml                 工作区（默认只构建 core + cli）
crates/align-core/         纯 Rust 核心库，无 GUI 依赖
  src/audio.rs             WAV 读写、交错缓冲、声道处理
  src/onset.rs             高通预加重 + 包络通量起振检测
  src/xcorr.rs             FFT 归一化互相关、多峰候选、全局/局部搜索
  src/anchor.rs            锚点模型、候选、置信度生命周期
  src/estimate.rs          RANSAC + IRLS 稳健拟合、连续分段拟合、残差
  src/map.rs               分段线性连续时间映射、正反求值、未知区间
  src/render.rs            加窗 sinc 重采样、统一输出长度与补零规则
  src/pipeline.rs          检测编排（全局粗延迟 → 局部精修 → 缺口保护）
  src/project.rs           工程文件、参考切换与缓存纪元
  src/synth.rs             已知延迟/漂移的合成测试素材
  src/verify.rs            估计误差、相位连续性、声道一致性
  src/hash.rs              无依赖 SHA-256（源文件指纹）
  tests/integration.rs     11 个可重复数值测试
crates/align-cli/          命令行：合成、建工程、对齐、导出、验证
crates/align-tauri/        Tauri v2 桌面应用（Rust 命令 + 原生 Web 前端）
  ui/                      TypeScript + Vite 前端
scripts/verify_synthetic.sh 一键可复现数值验证
```

## 时间映射约定

- 物理事件在参考时间 `t` 发生，被成员录音机打在自己时钟的
  `(t + delay) * ratio` 处（`ratio = 1 + drift`）。
- 拟合映射把成员时间映回共享工程时间：
  `project = member / ratio - delay`，即 `slope = 1/ratio`、`offset = -delay`。
- 分段映射在每段边界共享端点，因此跨录制中断的拼接保持连续；只有斜率变化，
  没有时间跳变。
- 没有可信锚点的区间保持 **Unknown**：预览时按最近已知斜率外推并在渲染报告
  里计数，绝不强制对齐覆盖演奏本身的差异。

## 锚点与候选

- 参考轨每个起振事件独立做一次局部互相关，锚点因此分布在整条录音上。
- 重复节拍会产生多个相关峰；每个锚点保留全部候选（滞后 + 归一化分数），
  可在 UI 选择或用 `pick_candidate` 命令切换，也可手动放置或标记未知。
- 稳健拟合先用 RANSAC（2 ms 容差）剔除选错峰的粗差，再用 IRLS 磨光；
  被剔除的锚点标记 `Rejected` 并保留残差，审计轨迹不丢失。

## 输出长度与补零规则（明确，不隐式）

- 所有分轨共享同一工程长度，取所有轨道映射末端的最大值，
  `frames = ceil(project_end * output_rate)`。
- 逆映射落在源素材之外的输出帧一律**补零**：前置部分计入 `head_pad_frames`，
  超出源末尾的计入 `tail_pad_frames`；边缘钳制与未知区间帧数分别统计。
- 重采样使用归一化 Hann 加窗 sinc 核（默认半长 32），同一条映射驱动全部声道，
  因此多声道素材不会产生声道间相位差。
- 原始 WAV 永不修改；`export-positions` 导出每个锚点的原始成员位置、映射后
  工程时间、滞后、残差与状态，未调整位置始终可查。

## 工程文件与参考切换

工程 JSON 保存参考轨、成员轨（含源文件 SHA-256）、全部锚点与候选、时间映射、
拟合摘要以及缺口元数据。`epoch` 在参考切换或源变更时递增；旧纪元的派生缓存
会被 `check_epoch` 显式拒绝（`StaleCache` 错误），切换参考会清空锚点与映射，
杜绝新旧缓存混用。

## 构建与测试

核心库与 CLI 只需要稳定 Rust 工具链：

```bash
cargo build -p align-core -p align-cli
cargo test -p align-core
cargo clippy -p align-core -p align-cli --all-targets
```

Tauri GUI 需要系统 webkit2gtk-4.1 / GTK3 前置依赖（见
<https://tauri.app/start/prerequisites/>），因此默认被工作区排除：

```bash
cd crates/align-tauri
npm --prefix ui install
npm --prefix ui run build
cargo build          # 在已安装 Tauri 系统依赖的主机上
```

## 可复现数值验证

```bash
./scripts/verify_synthetic.sh
```

脚本用 120 秒双声道素材（延迟 250 ms、漂移 150 ppm、参考缺段 40–45 s、
成员中断 80–83 s）跑完整流水线，并断言：

| 指标 | 典型实测值 | 阈值 |
| --- | --- | --- |
| 固定延迟误差 | 0.10 ms | ≤ 2 ms |
| 漂移误差 | 1.2 ppm | ≤ 25 ppm |
| 锚点 RMS 残差 | 0.05 ms | ≤ 2 ms |
| 声道间最大差 | 0 | < 1e-6 |

## CLI 用法

```bash
align-cli synthesize --out DIR [--delay S] [--drift-ppm P] [--channels N] \
                     [--ref-gap A:B] [--member-gap A:B]
align-cli init-project --name N --reference R.wav --member M.wav [--ref-gap A:B] \
                       [--member-gap A:B] --out project.json
align-cli align --project project.json --out aligned/
align-cli export-positions --project project.json --out positions.csv
align-cli verify-project --project project.json --truth synth/truth.json \
                         --aligned-dir aligned --out verification.json
```

`aligned/` 内除每条分轨外，还包含 `mix_preview.wav`（等增益混音）与
`before_after.wav`（原始成员轨 + 2 秒静音 + 对齐后成员轨），用于试听前后对比。
