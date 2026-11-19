"""Source-backed, navigable notes for material limits of this NRD integration."""
from pathlib import Path
import json
import html

root=Path(__file__).resolve().parents[1]
out=root/'build/falcor-nrd4'
run=json.loads((out/'run.json').read_text()); scores=json.loads((out/'metrics.json').read_text())
nrd='https://github.com/NVIDIA-RTX/NRD/blob/'+run['nrd_commit']
sample='https://github.com/NVIDIA-RTX/NRD-Sample/blob/35f5432ccbec3d234230b43a7216b40e65a58bdc'
falcor='https://github.com/NVIDIAGameWorks/Falcor/blob/9dc819c162b2070335c65060436041690b7937f8'
text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>NRD 4.18 庭院接入限制</title>
<style>body{{font:16px/1.7 system-ui;max-width:980px;margin:40px auto;padding:0 24px;color:#16212c}}nav,a{{color:#1762a6}}article{{border-top:1px solid #ccd4dc;padding:20px 0}}code{{overflow-wrap:anywhere}}small{{color:#526171}}</style>
<h1>NRD 4.18 庭院接入：结果解释边界</h1>
<p>检查对象：Falcor 8 + 官方 NRD 4.18.0 DXIL；NRD commit <code>{run['nrd_commit']}</code>。
参考采样模式：NRD-Sample <code>simplex</code>，commit <code>35f5432ccbec3d234230b43a7216b40e65a58bdc</code>。
保留了原 Falcor 的逐像素 1 spp 概率选瓣采样，640×360，静态相机。RELAX 三个独立实例，无 TAA/DLSS。</p>
<nav><a href="#coverage">采样覆盖限制</a> · <a href="#transmission">透射引导限制</a> · <a href="comparison.png">对比图</a> · <a href="metrics.json">指标</a></nav>
<article id="coverage"><h2>FAIL — 原有采样没有提供 NRD 的空间覆盖保证</h2>
<p><b>依据：采样契约与数据一致性。</b>当前实验保留 Falcor 的原始 BSDF 概率采样；
设置 AREA_5X5 并启用默认预滤波，仍不能保证每个有效邻域里两种反射分量都有样本。
这可能使稀疏镜面信号的距离重建与去噪不稳定；本次没有以覆盖失败统计图逐像素证明其视觉影响。</p>
<p>修正需要按概率范围与 Bayer 覆盖策略重新安排采样并保持估计器补偿一致。
这会改变现有噪声序列，不能无声地纳入本次“同一份输入”的对比。
验证时应在几何相容的邻域统计各瓣有效命中距离覆盖，再重新为所有滤波器生成同一份新输入。</p>
<p>证据：<a href="{nrd}/README.md#noisy-inputs">NRD 命中距离与概率选瓣输入要求</a>；
<a href="{sample}/Shaders/TraceOpaque.cs.hlsl#L130">官方示例的首瓣 Bayer/Weyl 选取</a>；
<a href="{sample}/Shaders/RaytracingShared.hlsli#L473">官方示例的概率范围</a>；
<a href="{falcor}/Source/RenderPasses/PathTracer/PathTracer.slang#L385">本次保留的 Falcor BSDF 采样调用</a>；
<a href="../../tools/falcor_nrd4_plugin/NRD4Pass.cpp">本地重建与预滤波设置</a>。</p></article>
<article id="transmission"><h2>FAIL — 透射信号缺少对应的有效替代表面引导</h2>
<p><b>依据：观测数据与引导一致性。</b>最终帧中，透射层所有像素的 Falcor 光程引导均无效，
因此这一路 RGB 能量 100% 走原样输出；反射层约 1.26% 的 RGB 能量也走此回退。
这解释了部分瓶杯区域仍保留原始噪声，不能据此评价正确引导下的 NRD 折射去噪。</p>
<p>现有二级层只转换了旧 Falcor 的光程与法线格式；它没有完整的 PSR 虚拟法线、曲率修正及相应命中距离。
修正需要为对应光路建立一致的替代表面数据，或使用独立透明材质渲染路径。
应先在平面镜、平板玻璃上验证引导与光路对应，再测试本场景的曲面瓶杯。</p>
<p>证据：<a href="metrics.json">本次 secondary_guide_coverage 测量</a>；
<a href="{nrd}/README.md#overview">官方关于透明材质支持边界的说明</a>；
<a href="{falcor}/Source/RenderPasses/PathTracer/PathTracerNRD.slang">继承的 Falcor 二级光路引导生成</a>；
<a href="../../tools/falcor_nrd4_plugin/NRD4Prepare.cs.slang">本地二级引导转换与范围外原样输出</a>。</p></article>
<small>本页按 nrd-integration-review 技能记录接入限制，不把内核运行成功或未使用的可选功能列为评审结论。
当前对比衡量这套已记录的输入适配，不能代表 NRD-Sample 的最高画质。</small></html>'''
(out/'integration-notes.html').write_text(text,encoding='utf-8')
print(out/'integration-notes.html')
