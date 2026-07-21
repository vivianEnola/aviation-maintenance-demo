from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any

from src.aviation_vision.runtime import configure_runtime

configure_runtime()

import streamlit as st

from src.aviation_vision.cloud_inbox import QueueItem, SupabaseInbox
from src.aviation_vision.config import load_yaml
from src.aviation_vision.images import ImageValidationError, validate_image_bytes
from src.aviation_vision.inference import (
    InferenceOutput,
    ModelUnavailableError,
    load_yolo_model,
    run_inference,
)


st.set_page_config(
    page_title="卫星图像航空辅助分析",
    page_icon=":material/satellite_alt:",
    layout="wide",
)


MODE_LABELS = {
    "auto": "Auto 智能路由",
    "general": "YOLOv8n 通用检测",
    "cloud": "云与台风云系检测",
    "airport": "机场周边环境检测",
    "runway": "跑道状态分检测割",
}

MODE_IMAGE_SIZES = {
    "auto": 640,
    "general": 640,
    "cloud": 640,
    "airport": 1024,
    "runway": 640,
}

PROJECT_ROOT = Path(__file__).resolve().parent
DEMO_IMAGES = {
    "云与涡旋分析样例": PROJECT_ROOT / "assets/samples/clouds.png",
    "机场周边分析样例": PROJECT_ROOT / "assets/samples/airport.png",
    "跑道状态分析样例": PROJECT_ROOT / "assets/samples/runway.png",
    "其它场景分析样例": PROJECT_ROOT / "assets/samples/other.png",
}


def _release_model(model: Any) -> None:
    try:
        model.model.cpu()
    except (AttributeError, RuntimeError):
        pass


@st.cache_resource(
    max_entries=1,
    show_spinner=False,
    on_release=_release_model,
)
def _cached_model(model_id: str, source: str) -> Any:
    return load_yolo_model(model_id, source)


@st.cache_resource(max_entries=2, show_spinner=False)
def _inbox_client(
    url: str,
    key: str,
    bucket: str,
    table: str,
) -> SupabaseInbox:
    return SupabaseInbox(url=url, key=key, bucket=bucket, table=table)


def _init_state() -> None:
    st.session_state.setdefault("output", None)
    st.session_state.setdefault("source_image", None)
    st.session_state.setdefault("source_name", None)
    st.session_state.setdefault("source_sha256", None)


def _clear_output() -> None:
    st.session_state.output = None
    st.session_state.source_image = None
    st.session_state.source_name = None
    st.session_state.source_sha256 = None


def _secret_group(name: str) -> dict[str, Any] | None:
    try:
        value = st.secrets.get(name)
    except (FileNotFoundError, KeyError):
        return None
    return dict(value) if value else None


def _run_bytes(
    content: bytes,
    *,
    filename: str,
    mode_id: str,
    confidence: float,
    iou: float,
    image_size: int,
) -> InferenceOutput:
    thresholds = load_yaml("thresholds.yaml")
    input_cfg = thresholds["input"]
    validated = validate_image_bytes(
        content,
        max_pixels=int(input_cfg["max_pixels"]),
        allowed_formats={str(item).upper() for item in input_cfg["allowed_formats"]},
    )
    output = run_inference(
        mode_id=mode_id,
        image=validated.image,
        confidence=confidence,
        iou=iou,
        image_size=image_size,
        thresholds=thresholds,
        rules=load_yaml("advisory_rules.yaml"),
        model_loader=_cached_model,
    )
    output.report.metadata.update(
        {
            "filename": filename,
            "sha256": validated.sha256,
            "image_format": validated.format,
            "width": validated.width,
            "height": validated.height,
        }
    )
    st.session_state.output = output
    st.session_state.source_image = validated.image
    st.session_state.source_name = filename
    st.session_state.source_sha256 = validated.sha256
    return output


def _image_png_bytes(image: Any) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _render_classification(output: InferenceOutput) -> None:
    classification = output.report.classification
    if classification is None:
        return
    st.subheader("Auto 分类结果")
    metrics = st.columns(3)
    metrics[0].metric("Top-1 类别", classification.label)
    metrics[1].metric("Top-1 置信度", f"{classification.confidence:.1%}")
    margin = classification.margin
    metrics[2].metric("类别差值", "—" if margin is None else f"{margin:.1%}")
    if classification.second_label is not None:
        st.caption(
            f"Top-2：{classification.second_label}，"
            f"置信度 {(classification.second_confidence or 0):.1%}"
        )


def _render_report(output: InferenceOutput) -> None:
    # 注入 CSS：强化 Label 标题感，缩小 Value
    st.markdown("""
        <style>
        /* Label：16px 加粗，深色，作为明显的标题 */
        div[data-testid="stMetricLabel"] > div {
            font-size: 16px !important;
            font-weight: 700 !important;
            color: #0F172A !important;
            margin-bottom: 2px !important;
        }
        /* Value：14px 常规，中灰色，作为次级内容 */
        div[data-testid="stMetricValue"] > div {
            font-size: 14px !important;
            font-weight: 400 !important;
            color: #475569 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    report = output.report
    source_image = st.session_state.source_image
    st.subheader("识别结果")
    metrics = st.columns(4)
    metrics[0].metric("执行模型", report.executed_model or "未路由")
    metrics[1].metric("任务", report.task or "—")
    metrics[2].metric("目标数量", len(report.objects))
    metrics[3].metric("总耗时", f"{report.processing_ms:.0f} ms")

    _render_classification(output)

    image_columns = st.columns(2)
    with image_columns[0].container(border=True, height="stretch"):
        st.subheader("原始图像")
        st.image(source_image, width="stretch")
    with image_columns[1].container(border=True, height="stretch"):
        st.subheader("分析图像")
        st.image(output.annotated_image, width="stretch")

    with st.container(border=True):
        st.subheader("综合分析")
        st.write(report.summary or "暂无分析结论。")
        guidance = [*report.knowledge, *report.recommendations]
        if guidance:
            st.subheader("**影响与处置建议**")
            for item in guidance:
                st.markdown(f"- {item}")
        for warning in report.warnings:
            st.warning(warning, icon=":material/warning:")

    if report.objects:
        rows = [
            {
                "类别": item.label,
                "置信度": item.confidence,
                "掩膜占图比例": item.mask_area_ratio,
                "边界框": None if item.xyxy is None else [round(v, 1) for v in item.xyxy],
            }
            for item in report.objects
        ]
        st.dataframe(
            rows,
            column_config={
                "置信度": st.column_config.ProgressColumn(
                    "置信度", min_value=0.0, max_value=1.0, format="percent"
                ),
                "掩膜占图比例": st.column_config.NumberColumn(format="percent"),
            },
            hide_index=True,
        )

    report_json = json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
    with st.container(horizontal=True):
        st.download_button(
            "下载 JSON 结果",
            data=report_json,
            file_name=f"{st.session_state.source_name or 'result'}.json",
            mime="application/json",
            icon=":material/download:",
        )
        st.download_button(
            "下载标注图",
            data=_image_png_bytes(output.annotated_image),
            file_name=f"{st.session_state.source_name or 'result'}_annotated.png",
            mime="image/png",
            icon=":material/image:",
        )


def _process_queue_item(
    client: SupabaseInbox,
    item: QueueItem,
    *,
    mode_id: str,
    confidence: float,
    iou: float,
    image_size: int,
) -> None:
    try:
        content = client.download(item)
        output = _run_bytes(
            content,
            filename=item.original_name,
            mode_id=mode_id,
            confidence=confidence,
            iou=iou,
            image_size=image_size,
        )
        client.mark_processed(item.id, result=output.report.to_dict())
    except Exception as exc:
        client.mark_failed(item.id, str(exc))
        raise


_init_state()
thresholds = load_yaml("thresholds.yaml")
defaults = thresholds["inference"]

st.title("卫星图像航空辅助分析")
st.caption("YOLO 多模型路由、云与机场环境检测、跑道状态分割")
st.info(
    "系统用于图像智能筛查与辅助研判。涉及航班运行、机场开放、适航或气象处置时，"
    "请由具备资质的人员结合现场检查和正式业务资料作出决定。",
    icon=":material/info:",
)

with st.sidebar:
    st.header("推理设置")
    mode_id = st.selectbox(
        "模型模式",
        options=list(MODE_LABELS),
        format_func=MODE_LABELS.get,
        key="mode_id",
    )
    confidence = st.slider(
        "置信度阈值",
        min_value=0.01,
        max_value=0.95,
        value=float(defaults["confidence"]),
        step=0.01,
        key=f"confidence_{mode_id}",
    )
    iou = st.slider(
        "IoU 阈值",
        min_value=0.10,
        max_value=0.90,
        value=float(defaults["iou"]),
        step=0.05,
    )
    image_size = st.select_slider(
        "模型输入尺寸",
        options=[320, 480, 640, 768, 1024],
        value=int(MODE_IMAGE_SIZES.get(mode_id, defaults["image_size"])),
        key=f"image_size_{mode_id}",
    )
    st.caption(
        f"当前模式建议输入尺寸为 {MODE_IMAGE_SIZES.get(mode_id, defaults['image_size'])}；"
        "较低阈值可提高召回率，但也会增加误报。"
    )

source_mode = st.segmented_control(
    "图片来源",
    options=["manual", "demo", "cloud"],
    default="manual",
    required=True,
    format_func={
        "manual": "手动上传",
        "demo": "分析样例",
        "cloud": "云端收件箱",
    }.get,
    key="source_mode",
    on_change=_clear_output,
)

if source_mode == "manual":
    with st.form("manual_inference"):
        uploaded = st.file_uploader(
            "上传图片",
            type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"],
            help="最大 20 MB；应用会校验真实图片格式和像素数量。",
        )
        submitted = st.form_submit_button(
            "开始分析", type="primary", icon=":material/play_arrow:"
        )
    if submitted:
        if uploaded is None:
            st.error("请先上传图片。", icon=":material/error:")
        else:
            try:
                with st.status("正在执行模型推理…", expanded=True) as status:
                    status.write("校验图片")
                    _run_bytes(
                        uploaded.getvalue(),
                        filename=uploaded.name,
                        mode_id=mode_id,
                        confidence=confidence,
                        iou=iou,
                        image_size=image_size,
                    )
                    status.update(label="分析完成", state="complete", expanded=False)
            except (ImageValidationError, ModelUnavailableError) as exc:
                st.error(str(exc), icon=":material/error:")
            except Exception as exc:
                st.exception(exc)
elif source_mode == "demo":
    with st.container(border=True):
        st.subheader("内置分析样例")
        demo_name = st.selectbox("选择样例", options=list(DEMO_IMAGES), key="demo_name")
        demo_path = DEMO_IMAGES[demo_name]
        st.image(str(demo_path), width=360)
        if st.button(
            "分析样例",
            type="primary",
            icon=":material/play_arrow:",
            key="analyze_demo",
        ):
            try:
                with st.status("正在执行模型推理…", expanded=True) as status:
                    status.write("载入内置样例")
                    _run_bytes(
                        demo_path.read_bytes(),
                        filename=demo_path.name,
                        mode_id=mode_id,
                        confidence=confidence,
                        iou=iou,
                        image_size=image_size,
                    )
                    status.update(label="分析完成", state="complete", expanded=False)
            except (ImageValidationError, ModelUnavailableError) as exc:
                st.error(str(exc), icon=":material/error:")
            except Exception as exc:
                st.exception(exc)
else:
    with st.container(border=True):
        st.subheader("本地监听器配置")
        st.caption("网页无法直接读取你的电脑目录；请在本地运行监听器，它会把新图片上传到当前云端收件箱。")
        with st.form("cloud_listener_setup", border=False):
            watch_folder = st.text_input("待监听的本地文件夹", placeholder="例如：D:/MMSSTV/Received", key="listener_watch_folder")
            listener_device = st.text_input("设备 ID", value="mmsstv-windows-01", key="listener_device_id")
            listener_saved = st.form_submit_button("生成监听配置", icon=":material/settings:")
        if listener_saved:
            if not watch_folder.strip():
                st.warning("请输入本地文件夹地址。", icon=":material/folder_off:")
            elif not listener_device.strip():
                st.warning("请输入设备 ID。", icon=":material/devices:")
            else:
                listener_config = (
                    f'watch_folder = "{watch_folder.strip().replace(chr(92), "/")}"\n'
                    f'device_id = "{listener_device.strip()}"\n'
                    'bucket = "mmsstv-images"\n'
                    'table = "image_queue"\n'
                    'scan_interval_seconds = 5\n'
                    'stable_wait_seconds = 2\n'
                    'extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]\n'
                )
                st.success("配置已生成。下载后保存为 local_uploader/uploader.toml。", icon=":material/check_circle:")
                st.download_button("下载 uploader.toml", data=listener_config, file_name="uploader.toml", mime="text/plain", icon=":material/download:")
                st.info("在本地 PowerShell 中设置 SUPABASE_URL 和 SUPABASE_SERVICE_ROLE_KEY，然后运行 start_folder_sync.bat。密钥不要放进配置文件或上传到网页。", icon=":material/terminal:")

    supabase_config = _secret_group("supabase")
    if not supabase_config:
        st.info(
            "云端收件箱尚未配置。请按 `.streamlit/secrets.example.toml` 设置 Supabase。",
            icon=":material/cloud_off:",
        )
    else:
        inbox = _inbox_client(
            str(supabase_config["url"]),
            str(supabase_config["key"]),
            str(supabase_config.get("bucket", "mmsstv-images")),
            str(supabase_config.get("table", "image_queue")),
        )
        device_id = st.text_input("设备 ID（留空表示全部）", key="inbox_device")
        auto_process = st.toggle(
            "发现新图片后自动解析",
            value=True,
            key="cloud_auto_process",
            help="每次刷新最多自动处理一张 pending 图片。",
        )

        @st.fragment(run_every="10s")
        def queue_panel() -> None:
            try:
                items = inbox.pending(device_id=device_id or None)
            except Exception as exc:
                st.error(f"读取云端收件箱失败：{exc}")
                return
            if not items:
                st.info("当前没有待处理图片。", icon=":material/inbox:")
                return
            if auto_process:
                selected = items[0]
                try:
                    with st.status(f"正在自动解析：{selected.original_name}", expanded=False):
                        _process_queue_item(
                            inbox,
                            selected,
                            mode_id=mode_id,
                            confidence=confidence,
                            iou=iou,
                            image_size=image_size,
                        )
                    st.toast(f"已完成：{selected.original_name}", icon=":material/check_circle:")
                    st.rerun()
                except Exception as exc:
                    st.error(f"自动解析失败：{exc}", icon=":material/error:")
                return
            selected_id = st.selectbox(
                "待处理图片",
                options=[item.id for item in items],
                format_func=lambda item_id: next(
                    item.original_name for item in items if item.id == item_id
                ),
                key="queue_item_id",
            )
            if st.button("分析所选图片", type="primary", icon=":material/play_arrow:"):
                selected = next(item for item in items if item.id == selected_id)
                try:
                    _process_queue_item(
                        inbox,
                        selected,
                        mode_id=mode_id,
                        confidence=confidence,
                        iou=iou,
                        image_size=image_size,
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"队列图片处理失败：{exc}")

        queue_panel()

if st.session_state.output is not None:
    _render_report(st.session_state.output)
