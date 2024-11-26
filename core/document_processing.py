import os
import zipfile
from typing import List, Dict
from pathlib import Path
from PyPDF2 import PdfReader
from docx import Document
import re


def extract_text_from_txt(file_path: str) -> str:
    """Извлечение текста из текстового файла."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    return "\n".join(page.extract_text() for page in reader.pages)


def extract_text_from_docx(file_path: str) -> str:
    """Извлечение текста из DOCX файла."""
    doc = Document(file_path)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])


def split_into_chunks(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """
    Разделение текста на контекстные окна фиксированного размера.

    :param text: Исходный текст.
    :param chunk_size: Размер окна.
    :param overlap: Количество перекрывающихся токенов между окнами.
    :return: Список текстовых кусочков.
    """
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", text)
    chunks = []
    current_chunk = []

    current_length = 0
    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap:]
            current_length = len(" ".join(current_chunk).split())

        current_chunk.append(sentence)
        current_length += sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def process_archive(archive_path: str, output_dir: str) -> List[Dict[str, str]]:
    """
    Обработка архива: распаковка, извлечение текста, разбиение на кусочки.

    :param archive_path: Путь к архиву.
    :param output_dir: Папка для временного хранения файлов.
    :return: Список словарей с кусочками текста.
    """
    # Распаковываем архив
    with zipfile.ZipFile(archive_path, 'r') as archive:
        archive.extractall(output_dir)

    # Список для хранения всех кусочков текста
    chunks_data = []

    # Проходимся по распакованным файлам
    for root, _, files in os.walk(output_dir):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            file_extension = Path(file_path).suffix.lower()

            try:
                if file_extension == ".txt":
                    text = extract_text_from_txt(file_path)
                elif file_extension == ".pdf":
                    text = extract_text_from_pdf(file_path)
                elif file_extension == ".docx":
                    text = extract_text_from_docx(file_path)
                else:
                    print(f"Unsupported file type: {file_name}")
                    continue

                # Разделяем текст на кусочки
                chunks = split_into_chunks(text)
                for idx, chunk in enumerate(chunks):
                    chunks_data.append({
                        "id": f"{Path(file_name).stem}_{idx}",
                        "file_path": file_path,
                        "chunk_text": chunk
                    })
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    return chunks_data
