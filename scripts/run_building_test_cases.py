#!/usr/bin/env python3
import argparse
import subprocess
import sys
import time
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent


def _safe_print(s: str) -> None:
    """
    安全地打印字符串，忽略或替换当前终端无法编码的特殊字符，
    避免在某些 Windows CMD 下因编码问题抛出 UnicodeEncodeError。
    """
    try:
        print(s)
    except UnicodeEncodeError:
        # 使用当前标准输出的编码进行编码，替换无法识别的字符，然后再解码回来打印
        encoding = sys.stdout.encoding or "utf-8"
        safe_str = s.encode(encoding, errors="replace").decode(encoding)
        print(safe_str)


TEST_CASES = [
    {
        "id": 1,
        "name": "双层紧凑住宅",
        "prompt": "生成一栋 2 层紧凑型城市住宅，总建筑面积约 170 平方米。1F 包含玄关、客厅、餐厅、开放式厨房、公共卫生间、储藏间；2F 包含主卧、次卧、儿童房、书房、两个卫生间和小型家庭活动区。要求客餐厨公共空间连通，卧室尽量靠外墙采光，厨房与餐厅相邻，卫生间与卧室保持合理距离。",
    },
    {
        "id": 2,
        "name": "三层社区服务中心",
        "prompt": "生成一栋 3 层社区服务中心，总建筑面积约 360 平方米。1F 是接待大厅、等候区、咨询室、无障碍卫生间和设备间；2F 是多功能活动室、阅览室、儿童活动室、办公室和储藏间；3F 是会议室、培训教室、管理办公室、茶水间和卫生间。要求公共功能在低层，办公和会议在高层，活动空间需要良好采光。",
    },
    {
        "id": 3,
        "name": "四层小型联合办公楼",
        "prompt": "生成一栋 4 层小型联合办公楼，总建筑面积约 520 平方米。1F 包含前台、大堂、咖啡休闲区、访客会议室和后勤用房；2F 和 3F 为开放办公区、小会议室、电话间、茶水间、卫生间；4F 为大型会议室、管理办公室、洽谈室和露台前厅。要求垂直分区清晰，会议空间靠近交通核心，办公区尽量获得外窗。",
    },
    {
        "id": 4,
        "name": "三层精品民宿",
        "prompt": "生成一栋 3 层精品民宿，总建筑面积约 420 平方米。1F 包含接待厅、餐厅、厨房、公共休息区、洗衣间和卫生间；2F 包含 4 间客房、公共起居区、布草间；3F 包含 3 间较大客房、观景休息室和储藏间。要求客房需要外窗，厨房与餐厅相邻，服务用房集中布置，公共空间与客房区分明确。",
    },
    {
        "id": 5,
        "name": "两层儿童学习中心",
        "prompt": "生成一栋 2 层儿童学习中心，总建筑面积约 260 平方米。1F 包含入口大厅、家长等候区、接待办公室、低龄教室、卫生间和储物间；2F 包含普通教室、美术教室、阅读室、教师办公室、会议室和卫生间。要求儿童教室采光好，等候区靠近入口，教师办公与教室保持便利联系，服务空间不要占据主要采光面。",
    },
]


def _format_duration(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(round(seconds % 60))
    return f"{m}m {s}s"


def _validate_out_dir_case(path: Path) -> bool:
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    out_root = (PROJECT_ROOT / "out").resolve()
    if str(path).startswith(str(out_root)):
        return True
    return False


def _parse_args(argv):
    p = argparse.ArgumentParser(description="批量运行 building 模式测试案例")
    p.add_argument("--case", type=str, help="指定案例编号，逗号分隔，例如 1,3,5")
    p.add_argument("--list", action="store_true", help="列出所有案例并退出")
    p.add_argument("--overwrite", action="store_true", help="允许清理并覆盖既有案例文件夹")
    p.add_argument("--skip-interior", action="store_true", help="跳过室内家具细化")
    p.add_argument("--model", type=str, default="gemini-2.5-pro", help="LLM 模型名")
    p.add_argument("--provider", type=str, default="gemini", choices=["openai", "gemini", "deepseek"], help="LLM 提供商")
    p.add_argument("--core", type=str, default="north", choices=["north", "center", "south"], help="核心筒位置")
    p.add_argument("--concurrency", type=int, default=8, help="室内家具并发数")
    return p.parse_args(argv)


def main(argv) -> int:
    args = _parse_args(argv)

    if args.list:
        _safe_print("=== Building 模式批量测试案例列表 ===\n")
        for case in TEST_CASES:
            _safe_print(f"Case {case['id']}: {case['name']}")
            _safe_print(f"  Prompt: {case['prompt'][:100]}...\n")
        return 0

    selected_ids = set()
    if args.case:
        for tok in args.case.strip().split(","):
            tok = tok.strip()
            if tok:
                try:
                    selected_ids.add(int(tok))
                except ValueError:
                    print(f" 忽略无效案例编号: {tok}", file=sys.stderr)

    cases = [c for c in TEST_CASES if (not selected_ids or c["id"] in selected_ids)]

    if not cases:
        print("没有选择任何测试案例", file=sys.stderr)
        return 1

    _safe_print("=== Building 模式批量测试 ===\n")
    _safe_print(f"选中案例数: {len(cases)}")
    _safe_print(f"默认参数: --model={args.model} --provider={args.provider} --core={args.core} --concurrency={args.concurrency}\n")

    results = []
    t0_total = time.time()

    for case in cases:
        case_id = case["id"]
        case_name = case["name"]
        case_prompt = case["prompt"]

        out_dir = PROJECT_ROOT / "out" / f"test_building_{case_id:02d}"

        if out_dir.exists() and args.overwrite:
            if not _validate_out_dir_case(out_dir):
                print(f"跳过 Case {case_id}: 输出目录 {out_dir} 不在 out/ 下，不安全删除", file=sys.stderr)
                results.append((case_id, case_name, False, 0.0, "不安全的目录"))
                continue
            _safe_print(f"清理既有案例: {out_dir.name}")
            shutil.rmtree(out_dir)

        out_dir.mkdir(parents=True, exist_ok=True)

        with open(out_dir / "case_prompt.txt", "w", encoding="utf-8") as f:
            f.write(case_prompt)

        cmd_parts = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "full_pipeline.py"),
            "-p", case_prompt,
            "-m", args.model,
            "--provider", args.provider,
            "-c", args.core,
            "--out-dir", str(out_dir),
            "--render-mode", "seg",
            "--seg-target", "refined",
            "--cad",
            "--concurrency", str(args.concurrency),
        ]
        if args.skip_interior:
            cmd_parts.append("--skip-interior")

        with open(out_dir / "run_command.txt", "w", encoding="utf-8") as f:
            f.write(" ".join(repr(p) for p in cmd_parts))

        _safe_print(f" 正在执行 Case {case_id}: {case_name} ...")
        t0 = time.time()
        success = False
        error_msg = ""
        try:
            with open(out_dir / "run.log", "w", encoding="utf-8", buffering=1) as log_file:
                ret = subprocess.run(
                    cmd_parts,
                    cwd=str(PROJECT_ROOT),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
            success = ret.returncode == 0
            if not success:
                error_msg = f"进程异常退出 (返回码: {ret.returncode})"
        except Exception as e:
            error_msg = f"脚本内部错误: {str(e)}"
            
        elapsed = time.time() - t0
        results.append((case_id, case_name, success, elapsed, error_msg))

    t_total = time.time() - t0_total

    _safe_print("\n" + "=" * 60)
    _safe_print("=== 批量测试执行报告 ===")
    _safe_print("=" * 60)
    success_count = 0
    for (case_id, case_name, success, elapsed, error_msg) in results:
        if success:
            success_count += 1
        status_text = "成功" if success else "失败"
        
        detail = f" Case {case_id} ({case_name}): {status_text} (耗时 {_format_duration(elapsed)})"
        if not success:
            detail += f" - {error_msg}. 详见 out/test_building_{case_id:02d}/run.log"
            
        _safe_print(detail)

    _safe_print("-" * 60)
    _safe_print(f"总计耗时: {_format_duration(t_total)} | 成功: {success_count} | 失败: {len(results)-success_count}")
    _safe_print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))