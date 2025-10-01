"""Telegram bot for filtering and reformatting credential lists.

This bot guides the user through a short conversation:
1. The user provides the current format of the lines in the input file
   (for example ``{mail}:{mailpass}:{username}:{password}``).
2. The user provides the desired output format
   (for example ``{username}:{password}``).
3. The user selects the maximum number of lines allowed in a single
   output file.
4. Finally, the user uploads a text file or provides a direct download
   link. The bot downloads, processes, and returns the filtered
   documents. If too many output files would be produced, they are
   grouped into archives. When the archives exceed Telegram's upload
   limit, the bot splits them into multiple volumes.

Set the ``TELEGRAM_BOT_TOKEN`` environment variable before running the
bot.
"""
from __future__ import annotations

import io
import logging
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
from urllib.parse import urlparse

import requests
from telegram import InputFile, Update
from telegram.constants import ChatAction
from telegram.ext import (ApplicationBuilder, CommandHandler,
                          ConversationHandler, ContextTypes, MessageHandler,
                          filters)

# Conversation states
SOURCE_FORMAT, TARGET_FORMAT, LINES_PER_FILE, AWAITING_FILE = range(4)

TOKEN_PATTERN = re.compile(r"\{([^{}]+)\}")
TELEGRAM_FILE_LIMIT = 49 * 1024 * 1024  # Reserve a little margin below 50MB.
MAX_PLAIN_FILES = 5  # If more files are produced, bundle them in archives.
FILENAME_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass
class ParsedFormat:
    """A helper structure describing a parsed format string."""

    raw: str
    tokens: Sequence[str]
    regex: re.Pattern[str]


class FormatError(ValueError):
    """Raised when a format string cannot be parsed or validated."""


@dataclass
class ProcessResult:
    """Store information about processed output files."""

    entries: List[Tuple[str, bytes]]
    skipped_lines: int

    @property
    def produced_files(self) -> int:
        return len(self.entries)


def extract_tokens(template: str) -> List[str]:
    """Return the ordered tokens that appear in ``template``."""

    tokens = [match.group(1).strip() for match in TOKEN_PATTERN.finditer(template)]
    if not tokens:
        raise FormatError("Формат должен содержать хотя бы один плейсхолдер в фигурных скобках.")
    return tokens


def parse_format(template: str) -> ParsedFormat:
    """Build a regular expression that can capture the fields in ``template``."""

    tokens = extract_tokens(template)
    pattern_parts: List[str] = []
    cursor = 0
    for match in TOKEN_PATTERN.finditer(template):
        literal = template[cursor:match.start()]
        if literal:
            pattern_parts.append(re.escape(literal))
        token = match.group(1).strip()
        if not token:
            raise FormatError("Имя плейсхолдера не может быть пустым.")
        # The non-greedy match allows separators to work correctly.
        pattern_parts.append(f"(?P<{token}>.+?)")
        cursor = match.end()
    literal = template[cursor:]
    if literal:
        pattern_parts.append(re.escape(literal))

    regex = re.compile("^" + "".join(pattern_parts) + "$")
    return ParsedFormat(raw=template, tokens=tokens, regex=regex)


def render_template(template: str, values: Dict[str, str]) -> str:
    """Render ``template`` by replacing each placeholder with ``values``."""

    def replacer(match: re.Match[str]) -> str:
        token = match.group(1).strip()
        if token not in values:
            raise KeyError(token)
        return values[token]

    return TOKEN_PATTERN.sub(replacer, template)


def transform_lines(
    lines: Iterable[str],
    source_format: ParsedFormat,
    target_format: str,
) -> Tuple[List[str], List[str]]:
    """Transform ``lines`` according to the provided formats.

    Returns a tuple ``(transformed, skipped)`` where ``transformed``
    contains the successfully converted lines and ``skipped`` stores the
    input lines that were ignored because they could not be parsed or
    lacked required tokens.
    """

    transformed: List[str] = []
    skipped: List[str] = []

    for line in lines:
        clean_line = line.rstrip("\n\r")
        if not clean_line:
            continue
        match = source_format.regex.match(clean_line)
        if not match:
            skipped.append(clean_line)
            continue
        values = match.groupdict()
        try:
            transformed_line = render_template(target_format, values)
        except KeyError:
            skipped.append(clean_line)
            continue
        transformed.append(transformed_line)
    return transformed, skipped


def chunk_sequence(sequence: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    """Yield consecutive chunks of ``size`` items."""

    for start in range(0, len(sequence), size):
        yield sequence[start : start + size]


def ensure_target_tokens(source: ParsedFormat, target: str) -> None:
    """Ensure that every token required by ``target`` exists in ``source``."""

    source_tokens = set(source.tokens)
    missing = [token for token in extract_tokens(target) if token not in source_tokens]
    if missing:
        raise FormatError(
            "В целевом формате используются неизвестные поля: " + ", ".join(sorted(set(missing)))
        )


def build_text_file(filename: str, lines: Sequence[str]) -> Tuple[str, bytes]:
    """Create an in-memory text file with ``lines`` separated by newlines."""

    buffer = io.StringIO()
    buffer.write("\n".join(lines))
    buffer.seek(0)
    return filename, buffer.read().encode("utf-8")


def package_archives(entries: Sequence[Tuple[str, bytes]]) -> List[Tuple[str, io.BytesIO]]:
    """Split ``entries`` into archives that respect Telegram's size limit."""

    packages: List[Tuple[str, io.BytesIO]] = []
    start = 0
    index = 1
    total = len(entries)

    while start < total:
        current_entries: List[Tuple[str, bytes]] = []
        end = start
        last_archive: io.BytesIO | None = None
        # Always include at least one file.
        while end < total:
            candidate_entries = current_entries + [entries[end]]
            archive = _build_zip_stream(candidate_entries)
            size = archive.getbuffer().nbytes
            if size <= TELEGRAM_FILE_LIMIT or not current_entries:
                current_entries = candidate_entries
                end += 1
                last_archive = archive
            else:
                break
        if last_archive is None:
            raise RuntimeError("Не удалось подготовить архив для отправки.")
        packages.append((f"processed_part_{index:03d}.zip", last_archive))
        start += len(current_entries)
        index += 1
    return packages


def _build_zip_stream(entries: Sequence[Tuple[str, bytes]]) -> io.BytesIO:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for name, data in entries:
            zip_file.writestr(name, data)
    stream.seek(0)
    return stream


def is_valid_url(text: str) -> bool:
    parsed = urlparse(text)
    return all([parsed.scheme in {"http", "https"}, parsed.netloc])


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        "Привет! Отправьте текущий формат данных, например {mail}:{password}."
    )
    return SOURCE_FORMAT


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.clear()
    await update.message.reply_text("Обработка отменена. Вы можете начать заново командой /start.")
    return ConversationHandler.END


async def handle_source_format(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    raw_format = update.message.text.strip()
    try:
        parsed = parse_format(raw_format)
    except FormatError as exc:
        await update.message.reply_text(f"Ошибка формата: {exc}")
        return SOURCE_FORMAT

    context.user_data["source_format"] = parsed
    await update.message.reply_text(
        "Теперь отправьте желаемый формат строк, например {username}:{password}."
    )
    return TARGET_FORMAT


async def handle_target_format(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    target = update.message.text.strip()
    source_format: ParsedFormat = context.user_data["source_format"]
    try:
        ensure_target_tokens(source_format, target)
    except FormatError as exc:
        await update.message.reply_text(f"Ошибка формата: {exc}")
        return TARGET_FORMAT

    context.user_data["target_format"] = target
    await update.message.reply_text("Сколько строк должно быть в одном выходном файле?")
    return LINES_PER_FILE


async def handle_lines_per_file(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text.strip()
    if not text.isdigit() or int(text) <= 0:
        await update.message.reply_text("Введите положительное число.")
        return LINES_PER_FILE

    context.user_data["lines_per_file"] = int(text)
    await update.message.reply_text(
        "Отправьте текстовый файл для обработки или ссылку на скачивание."
    )
    return AWAITING_FILE


async def handle_file_or_link(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    source_format: ParsedFormat = context.user_data["source_format"]
    target_format: str = context.user_data["target_format"]
    lines_per_file: int = context.user_data["lines_per_file"]

    try:
        if update.message.document:
            result = await _handle_document(
                update, context, source_format, target_format, lines_per_file
            )
        elif update.message.text and is_valid_url(update.message.text.strip()):
            result = await _handle_url(
                update, context, source_format, target_format, lines_per_file
            )
        else:
            await update.message.reply_text(
                "Отправьте файл в формате .txt или прямую ссылку на загрузку."
            )
            return AWAITING_FILE
    except Exception as exc:  # noqa: BLE001
        logging.exception("Error while processing input")
        await update.message.reply_text(f"Произошла ошибка: {exc}")
        return ConversationHandler.END

    produced_files = await _finalize_processing(update, context, result)

    if produced_files:
        await update.message.reply_text(
            f"Готово! Обработано файлов: {produced_files}. Для новой обработки отправьте /start."
        )
    else:
        await update.message.reply_text("Для новой обработки отправьте /start.")

    context.user_data.clear()
    return ConversationHandler.END


async def _handle_document(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    source_format: ParsedFormat,
    target_format: str,
    lines_per_file: int,
) -> ProcessResult:
    document = update.message.document
    if document.file_size and document.file_size > TELEGRAM_FILE_LIMIT:
        raise ValueError("Входной файл превышает ограничение Telegram (≈49 МБ).")

    await update.message.reply_chat_action(ChatAction.UPLOAD_DOCUMENT)
    telegram_file = await document.get_file()
    with tempfile.TemporaryDirectory() as tmp_dir:
        filename = document.file_name or "input.txt"
        local_path = Path(tmp_dir) / filename
        await telegram_file.download_to_drive(custom_path=str(local_path))
        return _process_local_file(
            local_path, source_format, target_format, lines_per_file
        )


async def _handle_url(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    source_format: ParsedFormat,
    target_format: str,
    lines_per_file: int,
) -> ProcessResult:
    url = update.message.text.strip()
    await update.message.reply_chat_action(ChatAction.TYPING)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(response.content)
        tmp_path = Path(tmp_file.name)
    try:
        return _process_local_file(tmp_path, source_format, target_format, lines_per_file)
    finally:
        tmp_path.unlink(missing_ok=True)


def _process_local_file(
    file_path: Path,
    source_format: ParsedFormat,
    target_format: str,
    lines_per_file: int,
) -> ProcessResult:
    if zipfile.is_zipfile(file_path):
        return _process_zip_archive(file_path, source_format, target_format, lines_per_file)

    with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
        return _process_stream(
            handle,
            file_path.name,
            source_format,
            target_format,
            lines_per_file,
            {},
        )


def _process_zip_archive(
    archive_path: Path,
    source_format: ParsedFormat,
    target_format: str,
    lines_per_file: int,
) -> ProcessResult:
    entries: List[Tuple[str, bytes]] = []
    skipped_total = 0
    name_usage: Dict[str, int] = {}
    has_files = False

    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            has_files = True
            with archive.open(info) as raw_stream:
                with io.TextIOWrapper(raw_stream, encoding="utf-8", errors="ignore") as text_stream:
                    result = _process_stream(
                        text_stream,
                        info.filename,
                        source_format,
                        target_format,
                        lines_per_file,
                        name_usage,
                    )
                entries.extend(result.entries)
                skipped_total += result.skipped_lines

    if not has_files:
        raise ValueError("Архив не содержит файлов для обработки.")

    return ProcessResult(entries=entries, skipped_lines=skipped_total)


def _process_stream(
    stream: Iterable[str],
    source_name: str,
    source_format: ParsedFormat,
    target_format: str,
    lines_per_file: int,
    name_usage: Dict[str, int],
) -> ProcessResult:
    transformed, skipped = transform_lines(stream, source_format, target_format)

    if not transformed:
        return ProcessResult(entries=[], skipped_lines=len(skipped))

    chunks = list(chunk_sequence(transformed, lines_per_file))
    base_name = _allocate_base_name(source_name, name_usage)
    multi_part = len(chunks) > 1
    entries: List[Tuple[str, bytes]] = []
    for index, chunk in enumerate(chunks, start=1):
        suffix = f"_part_{index:03d}" if multi_part else ""
        filename = f"{base_name}{suffix}.txt"
        entries.append(build_text_file(filename, chunk))

    return ProcessResult(entries=entries, skipped_lines=len(skipped))


def _allocate_base_name(raw_name: str, name_usage: Dict[str, int]) -> str:
    base = _sanitize_basename(raw_name)
    count = name_usage.get(base, 0)
    name_usage[base] = count + 1
    if count == 0:
        return base
    return f"{base}_{count + 1:02d}"


def _sanitize_basename(raw_name: str) -> str:
    stem = Path(raw_name).stem or "processed"
    sanitized = FILENAME_SANITIZE_RE.sub("_", stem).strip("._")
    return sanitized or "processed"


async def _finalize_processing(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    result: ProcessResult,
) -> int:
    if not result.entries:
        await update.message.reply_text(
            "Не удалось преобразовать ни одной строки. Проверьте форматы и попробуйте снова."
        )
        if result.skipped_lines:
            await update.message.reply_text(
                f"Пропущено строк: {result.skipped_lines}. Они не совпали с исходным форматом."
            )
        return 0

    await _deliver_results(update, context, result.entries)

    if result.skipped_lines:
        await update.message.reply_text(
            f"Пропущено строк: {result.skipped_lines}. Они не совпали с исходным форматом."
        )

    return result.produced_files


async def _deliver_results(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    entries: Sequence[Tuple[str, bytes]],
) -> None:
    chat_id = update.effective_chat.id
    application = context.application

    if len(entries) == 1 and len(entries[0][1]) <= TELEGRAM_FILE_LIMIT:
        filename, data = entries[0]
        await application.bot.send_document(chat_id=chat_id, document=InputFile(io.BytesIO(data), filename))
        return

    if len(entries) <= MAX_PLAIN_FILES:
        for filename, data in entries:
            if len(data) > TELEGRAM_FILE_LIMIT:
                break
        else:
            for filename, data in entries:
                await application.bot.send_document(
                    chat_id=chat_id,
                    document=InputFile(io.BytesIO(data), filename),
                )
            return

    archives = package_archives(entries)
    for filename, archive in archives:
        await application.bot.send_document(chat_id=chat_id, document=InputFile(archive, filename))


def build_application() -> "telegram.ext.Application":
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("Переменная окружения TELEGRAM_BOT_TOKEN не задана.")

    application = ApplicationBuilder().token(token).build()

    conversation = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            SOURCE_FORMAT: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_source_format)],
            TARGET_FORMAT: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_target_format)],
            LINES_PER_FILE: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_lines_per_file)],
            AWAITING_FILE: [
                MessageHandler(filters.Document.ALL, handle_file_or_link),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_file_or_link),
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    application.add_handler(conversation)
    application.add_handler(CommandHandler("cancel", cancel))

    return application


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    application = build_application()
    application.run_polling()


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        pass
