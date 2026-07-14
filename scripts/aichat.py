#!/usr/bin/env python3
"""
AIChat - AI chat in terminal with OpenAI-compatible API.

Usage:
    python aichat.py              # start interactive chat
    python aichat.py --config     # show current config
    python aichat.py --set        # interactive config setup
    python aichat.py -h           # show help

Dependencies: openai, rich, prompt_toolkit
    pip install openai rich prompt-toolkit
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Union

from openai import APIError, APIConnectionError, AuthenticationError, OpenAI, Timeout
from prompt_toolkit import prompt
from prompt_toolkit.styles import Style as PtStyle
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.theme import Theme
from rich.status import Status


# ──────────────────────────────────────────────
#  byksdk  – embedded plugin SDK
# ──────────────────────────────────────────────

def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=path.parent, encoding="utf-8"
    ) as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        tmp = Path(f.name)
    tmp.replace(path)


@dataclass(slots=True)
class StateStore:
    path: Path

    def load(self) -> dict[str, Any]:
        return _read_json(self.path)

    def save(self, data: dict[str, Any]) -> dict[str, Any]:
        _write_json(self.path, data)
        return data

    def get(self, key: str, default: Any = None) -> Any:
        return self.load().get(key, default)

    def set(self, key: str, value: Any) -> dict[str, Any]:
        data = self.load()
        data[key] = value
        return self.save(data)

    def update(self, values: dict[str, Any]) -> dict[str, Any]:
        data = self.load()
        data.update(values)
        return self.save(data)

    def delete(self, key: str) -> dict[str, Any]:
        data = self.load()
        data.pop(key, None)
        return self.save(data)

    def clear(self) -> dict[str, Any]:
        return self.save({})


ROOT_DIR = Path.home() / ".byk"
STATE_DIR = ROOT_DIR / "state"
PLUGINS_DIR = ROOT_DIR / "plugins"
LOGS_DIR = ROOT_DIR / "logs"
RUNTIME_DIR = ROOT_DIR / "runtime"
CACHE_DIR = ROOT_DIR / "cache"
PLUGINS_LOG_FILE = LOGS_DIR / "plugins.log"


def get_logger(name: str | None = None) -> logging.Logger:
    if name is None:
        logger_name = "byk"
        log_file = PLUGINS_LOG_FILE
    else:
        logger_name = f"byk.{name}"
        log_file = PLUGINS_DIR / name / f"{name}.log"

    logger = logging.getLogger(logger_name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


@dataclass
class _AppContext:
    logger: logging.Logger = field(default_factory=get_logger)

    def store(self, name: str = "plugins.state") -> StateStore:
        return StateStore(STATE_DIR / f"{name}.json")


@dataclass
class PluginContext:
    name: str
    home: Path
    logger: logging.Logger
    app: _AppContext = field(default_factory=_AppContext)

    def state(self, name: str = "state") -> StateStore:
        return StateStore(self.home / f"{name}.json")


def plugin(name: str) -> PluginContext:
    home = PLUGINS_DIR / name
    return PluginContext(name=name, home=home, logger=get_logger(name))


# ──────────────────────────────────────────────
#  AI service layer
# ──────────────────────────────────────────────

JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class ChatRequest:
    messages: List[JsonDict]
    model: str
    api_key: str
    api_url: str
    stream: bool = False
    timeout: int = 30
    extra_body: Dict[str, Any] = None


class AIServiceError(RuntimeError):
    @classmethod
    def network_error(cls):
        return cls("网络错误")

    @classmethod
    def api_key_error(cls):
        return cls("API Key 错误")

    @classmethod
    def backend_error(cls, detail: str = ""):
        msg = "后端逻辑错误"
        if detail:
            msg += f": {detail}"
        return cls(msg)


class AIService:
    def chat(self, req: ChatRequest) -> Union[JsonDict, Iterable[JsonDict]]:
        if not req.api_key:
            raise AIServiceError.api_key_error()

        try:
            base = (
                req.api_url.replace("/chat/completions", "")
                if "/chat/completions" in req.api_url
                else req.api_url
            )
            client = OpenAI(api_key=req.api_key, base_url=base, timeout=req.timeout)

            params: dict = {
                "model": req.model,
                "messages": req.messages,
                "stream": req.stream,
            }
            if req.extra_body:
                params["extra_body"] = req.extra_body

            if req.stream:
                return self._stream_chunks(client, params)
            else:
                return self._parse_response(client, params)

        except AuthenticationError:
            raise AIServiceError.api_key_error()
        except (APIConnectionError, Timeout):
            raise AIServiceError.network_error()
        except APIError as e:
            raise AIServiceError.backend_error(str(e))
        except Exception as e:
            raise AIServiceError.backend_error(str(e))

    def _parse_response(self, client: OpenAI, params: dict) -> JsonDict:
        try:
            response = client.chat.completions.create(**params)
            return response.model_dump()
        except (AuthenticationError, APIConnectionError, Timeout, APIError):
            raise
        except Exception as e:
            raise AIServiceError.backend_error(f"响应解析失败: {e}")

    def _stream_chunks(self, client: OpenAI, params: dict) -> Generator[JsonDict, None, None]:
        try:
            stream = client.chat.completions.create(**params)
            for chunk in stream:
                yield chunk.model_dump()
        except (AuthenticationError, APIConnectionError, Timeout, APIError):
            raise
        except Exception as e:
            raise AIServiceError.backend_error(f"流式读取失败: {e}")


def extract_assistant_reply(response: JsonDict) -> str:
    return response["choices"][0]["message"]["content"]


# ──────────────────────────────────────────────
#  streaming Markdown renderer
# ──────────────────────────────────────────────

class MarkdownRenderer:
    def __init__(self):
        self.buffer = ""
        self.state = "NORMAL"
        self.status: Status | None = None
        self.block_lines: list[str] = []
        self.full_reply = ""
        custom_theme = Theme({
            "markdown.h1": "bold magenta",
            "markdown.h2": "bold magenta",
        })
        self.console = Console(theme=custom_theme)

    def _stop_status(self):
        if self.status:
            self.status.stop()
            self.status = None

    def _render_header(self, line: str) -> bool:
        stripped_line = line.strip()
        header_match = re.match(r"^(#+)\s*(.*)", stripped_line)
        if not header_match:
            return False
        level = len(header_match.group(1))
        title = header_match.group(2).strip()
        title = re.sub(r"\*\*(.*?)\*\*", r"\1", title)
        level = min(max(level, 1), 6)
        header_colors = {
            1: "magenta", 2: "cyan", 3: "green", 4: "yellow",
            5: "bright_blue", 6: "bright_black",
        }
        color = header_colors.get(level, "white")
        if level == 1:
            self.console.print(f"\n[bold white on {color}] {title.upper()} [/bold white on {color}]")
        elif level == 2:
            self.console.print(f"\n[bold {color}]┃ {title}[/bold {color}]")
        elif level == 3:
            self.console.print(f"\n[bold {color}]◆ {title}[/bold {color}]")
        else:
            prefixes = {4: "◇", 5: "◈", 6: "○"}
            prefix = prefixes.get(level, "•")
            self.console.print(f"\n[bold {color}]{prefix} {title}[/bold {color}]")
        return True

    def _flush_block(self):
        if not self.block_lines:
            return
        self._stop_status()
        content = "\n".join(self.block_lines)

        if self.state == "CODE":
            lines = content.split("\n", 1)
            if len(lines) > 1 and lines[0].isalnum():
                lang, code = lines[0], lines[1]
            else:
                lang, code = "", content
            syntax = Syntax(
                code.strip(), lang or "text",
                theme="monokai", background_color="default",
                word_wrap=True, indent_guides=False,
            )
            panel = Panel(
                syntax, border_style="bright_blue", expand=False,
                title=lang if lang else None, title_align="right",
            )
            self.console.print(panel)
        elif self.state in ("TABLE", "LIST", "NORMAL"):
            remaining_lines: list[str] = []
            for line in self.block_lines:
                if line.strip().startswith("#") and self.state != "CODE":
                    if remaining_lines:
                        self.console.print(Markdown("\n".join(remaining_lines)))
                        remaining_lines = []
                    if self._render_header(line):
                        continue
                remaining_lines.append(line)
            if remaining_lines:
                self.console.print(Markdown("\n".join(remaining_lines)))
        self.block_lines = []
        self.state = "NORMAL"

    def render(self, chunks) -> str:
        first_chunk = True
        for chunk in chunks:
            if first_chunk:
                self.status = Status("[bold blue]正在生成响应...[/bold blue]", spinner="dots")
                self.status.start()
                first_chunk = False
            delta = chunk["choices"][0]["delta"].get("content", "")
            if not delta:
                continue
            self.full_reply += delta
            self.buffer += delta

            while "```" in self.buffer:
                if self.state != "CODE":
                    parts = self.buffer.split("```", 1)
                    if parts[0].strip() or self.block_lines:
                        if parts[0].strip():
                            self.block_lines.extend(parts[0].split("\n"))
                        self._flush_block()
                    self.state = "CODE"
                    self.buffer = parts[1]
                    self.status = Status("[bold blue]正在生成代码...[/bold blue]", spinner="dots")
                    self.status.start()
                else:
                    parts = self.buffer.split("```", 1)
                    self.block_lines.append(parts[0])
                    self._flush_block()
                    self.state = "NORMAL"
                    self.buffer = parts[1]

            if "\n" in self.buffer:
                lines = self.buffer.split("\n")
                finished_lines = lines[:-1]
                self.buffer = lines[-1]
                for line in finished_lines:
                    stripped = line.strip()
                    if not stripped and self.state == "NORMAL":
                        if self.block_lines:
                            self._flush_block()
                        continue
                    is_header = stripped.startswith("#")
                    is_list = (
                        stripped.startswith(("-", "*", "+"))
                        or (
                            stripped
                            and stripped[0].isdigit()
                            and (". " in stripped[:4] or " " in stripped[:4])
                        )
                        or stripped.startswith(("[ ]", "[x]", "[X]"))
                    )
                    is_table = "|" in stripped
                    if self.state == "CODE":
                        new_state = "CODE"
                    elif is_header:
                        new_state = "NORMAL"
                    elif is_table:
                        new_state = "TABLE"
                    elif is_list:
                        new_state = "LIST"
                    else:
                        new_state = "NORMAL"
                    if new_state != self.state:
                        if self.state != "NORMAL":
                            self._flush_block()
                        self.state = new_state
                        if self.state == "TABLE":
                            self.status = Status("[bold blue]正在生成表格...[/bold blue]", spinner="dots")
                            self.status.start()
                        elif self.state == "LIST":
                            self.status = Status("[bold blue]正在生成列表...[/bold blue]", spinner="dots")
                            self.status.start()
                    self.block_lines.append(line)
        if self.buffer:
            self.block_lines.append(self.buffer)
        self._flush_block()
        return self.full_reply


def print_streaming_chunks(chunks) -> str:
    return MarkdownRenderer().render(chunks)


def render_non_streaming_reply(text: str) -> None:
    console = Console()
    console.print(Panel(Markdown(text), border_style="bright_blue"))


# ──────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────

DEFAULT_CONFIG = {
    "model": "deepseek-v4-flash",
    "api_url": "https://api.deepseek.com/v1/chat/completions",
    "api_key": None,
    "stream": True,
    "rich": True,
    "extra_body": None,
}

SYSTEM_PROMPT = (
    "You are a helpful assistant. Respond in plain text suitable for a console environment. "
    "Avoid using Markdown, code blocks, or any rich formatting. "
    "Use simple line breaks and spaces for alignment."
)
SYSTEM_PROMPT_RICH = (
    "You are a helpful assistant. Respond using standard Markdown. "
    "Use code blocks for code, bold for emphasis, and lists where appropriate. "
    "Keep your responses concise and suitable for a terminal environment."
)


def _print_plain_chunks(chunks) -> str:
    console = Console()
    reply = ""
    console.print("[bold blue]AI:[/bold blue] ", end="")
    for chunk in chunks:
        delta = chunk["choices"][0]["delta"].get("content", "")
        if delta:
            print(delta, end="", flush=True)
            reply += delta
    print()
    return reply


_pt_style = PtStyle.from_dict({"prompt": "bold #ffff00"})


def _interactive_setup(state_config: dict) -> bool:
    console = Console()
    console.print()
    console.print("⚙️  [bold cyan]AI 配置设置[/bold cyan]")
    console.print("[dim]按 Enter 使用方括号中的默认值，输入新值后按 Enter 更新。[/dim]\n")

    changed = False

    # Model
    current = state_config.get("model", DEFAULT_CONFIG["model"])
    value = prompt(
        [("class:prompt", f"  Model [{current}]: ")],
        default=str(current), style=_pt_style,
    ).strip()
    if value and value != str(current):
        state_config["model"] = value
        changed = True

    # API Key
    has_key = bool(state_config.get("api_key"))
    if has_key:
        console.print(
            f'  [bold yellow]API Key:[/bold yellow] [dim][{"*" * 12}][/dim]  '
            f'[dim](按 Enter 保持不变)[/dim]'
        )
    else:
        console.print("  [bold yellow]API Key:[/bold yellow] [dim][(未设置)][/dim]")
    value = prompt(
        [("class:prompt", "  > ")],
        is_password=True, style=_pt_style,
    ).strip()
    if value:
        state_config["api_key"] = value
        changed = True

    # API URL
    current = state_config.get("api_url", DEFAULT_CONFIG["api_url"])
    value = prompt(
        [("class:prompt", f"  API URL [{current}]: ")],
        default=str(current), style=_pt_style,
    ).strip()
    if value and value != str(current):
        state_config["api_url"] = value
        changed = True

    # Stream
    current = state_config.get("stream", DEFAULT_CONFIG["stream"])
    if isinstance(current, str):
        current = current.lower() in ("1", "true")
    current_bool = bool(current)
    suffix = "[Y/n]" if current_bool else "[y/N]"
    answer = prompt(
        [("class:prompt",
          f'  Stream（流式输出，当前：{"ON" if current_bool else "OFF"}） {suffix}: ')],
        style=_pt_style,
    ).strip().lower()
    value = current_bool if not answer else answer in ("y", "yes")
    if value != current_bool:
        state_config["stream"] = value
        changed = True

    # Rich
    current = state_config.get("rich", DEFAULT_CONFIG["rich"])
    if isinstance(current, str):
        current = current.lower() in ("1", "true")
    current_bool = bool(current)
    suffix = "[Y/n]" if current_bool else "[y/N]"
    answer = prompt(
        [("class:prompt",
          f'  Rich（富文本渲染，当前：{"ON" if current_bool else "OFF"}） {suffix}: ')],
        style=_pt_style,
    ).strip().lower()
    value = current_bool if not answer else answer in ("y", "yes")
    if value != current_bool:
        state_config["rich"] = value
        changed = True

    # Extra body
    current = state_config.get("extra_body")
    current_display = json.dumps(current, ensure_ascii=False) if current else ""
    value = prompt(
        [("class:prompt",
          f"  Extra body（JSON，可选） [{current_display}]: " if current_display
          else "  Extra body（JSON，可选）: ")],
        default=current_display, style=_pt_style,
    ).strip()
    if value:
        try:
            parsed = json.loads(value)
            state_config["extra_body"] = parsed
            changed = True
        except json.JSONDecodeError:
            console.print("  [yellow]⚠ JSON 格式无效，保持当前值不变。[/yellow]")
    elif current is not None:
        state_config["extra_body"] = None
        changed = True

    console.print()
    return changed


def _chat_loop(ctx: PluginContext):
    console = Console()
    store = ctx.state()
    config = store.load()
    if not config:
        config = DEFAULT_CONFIG.copy()
        store.save(config)

    service = AIService()
    system_prompt = SYSTEM_PROMPT_RICH if config.get("rich") else SYSTEM_PROMPT
    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    model = config.get("model", "unknown")
    rich_on = bool(config.get("rich", True))
    stream_on = bool(config.get("stream", True))

    rich_label = "[green]ON[/green]" if rich_on else "[dim]OFF[/dim]"
    stream_label = "[green]ON[/green]" if stream_on else "[dim]OFF[/dim]"

    sep = "─" * 50
    console.print(f"[cyan]{sep}[/cyan]")
    console.print(
        f"[cyan]Model:[/cyan] [yellow]{model}[/yellow]  "
        f"[cyan]Rich:[/cyan] {rich_label}  "
        f"[cyan]Stream:[/cyan] {stream_label}"
    )
    console.print("[dim]输入 exit 退出 | Ctrl+C 中断[/dim]")
    console.print(f"[cyan]{sep}[/cyan]")

    while True:
        try:
            user_input = prompt([("class:prompt", "You: ")]).strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[cyan]Chat ended.[/cyan]")
            break

        if user_input.lower() == "exit":
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        req = ChatRequest(
            messages=messages,
            model=config["model"],
            api_key=config["api_key"],
            api_url=config["api_url"],
            stream=bool(config["stream"]),
            extra_body=config.get("extra_body"),
        )

        try:
            if config.get("rich"):
                if req.stream:
                    resp_or_chunks = service.chat(req)
                    reply = print_streaming_chunks(resp_or_chunks)
                else:
                    with Status("[bold blue]正在思考...[/bold blue]", spinner="dots"):
                        resp_or_chunks = service.chat(req)
                    reply = extract_assistant_reply(resp_or_chunks)
                    render_non_streaming_reply(reply)
            else:
                if not req.stream:
                    console.print("[blue]AI:[/blue] 正在思考...", end="")
                resp_or_chunks = service.chat(req)
                if req.stream:
                    reply = _print_plain_chunks(resp_or_chunks)
                else:
                    reply = extract_assistant_reply(resp_or_chunks)
                    print("\r", end="")
                    console.print(f"[blue]AI:[/blue] {reply}")

            messages.append({"role": "assistant", "content": reply})

        except AIServiceError as e:
            console.print(f"[red]Error: {e}[/red]")
            messages.pop()
        except Exception as e:
            console.print(f"[red]Unknown error: {e}[/red]")
            messages.pop()


def main():
    parser = argparse.ArgumentParser(
        prog="aichat",
        description="AI chat in terminal with OpenAI-compatible API",
    )
    parser.add_argument(
        "--config", "-c", action="store_true",
        help="show config and exit",
    )
    parser.add_argument(
        "--set", "-S", action="store_true", dest="set_",
        help="interactive configuration setup",
    )
    args = parser.parse_args()

    console = Console()
    ctx = plugin("ai-chat")
    store = ctx.state()
    state_config = store.load()
    if not state_config:
        state_config = DEFAULT_CONFIG.copy()

    if args.config:
        console.print("[bold cyan]AI Configuration:[/bold cyan]")
        for k, v in state_config.items():
            display = v
            if k == "api_key" and v:
                display = "*" * 12
            elif k == "extra_body" and v:
                display = json.dumps(v, ensure_ascii=False)
            console.print(f"  [cyan]{k}[/cyan]: [yellow]{display}[/yellow]")
        console.print(f"\n[dim]Config file: {store.path}[/dim]")
        return

    if args.set_:
        changed = _interactive_setup(state_config)
        if changed:
            store.save(state_config)
            console.print("[green]✅ 配置已保存。[/green]")
        else:
            console.print("[dim]配置未变更。[/dim]")
        console.print(f"[dim]Config file: {store.path}[/dim]\n")
        return

    if not state_config.get("api_key"):
        console.print("[yellow]⚠ API Key 未配置，进入交互式设置...[/yellow]")
        _interactive_setup(state_config)
        store.save(state_config)

        if not state_config.get("api_key"):
            console.print("[red]❌ 未设置 API Key，无法启动聊天。[/red]")
            sys.exit(1)

        console.print("[green]✅ 配置已保存，启动聊天...[/green]\n")

    _chat_loop(ctx)


if __name__ == "__main__":
    main()
