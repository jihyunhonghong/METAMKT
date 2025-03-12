from pymilvus import connections, utility
import re
import pdfplumber
import warnings
import io
import concurrent.futures
import json
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from pymilvus import FieldSchema, CollectionSchema, DataType, Collection
from concurrent.futures import ThreadPoolExecutor
from langchain.embeddings.openai import OpenAIEmbeddings

# 경고 메시지 무시 및 환경 변수 로드
warnings.filterwarnings("ignore")
load_dotenv()

# Milvus "sk_doc_test" 데이터베이스에 연결 (alias와 db_name 지정)
connections.connect(
    alias="sk_doc_test",
    host="49.50.175.45",
    port="19530",
    db_name="sk_doc_test"
)
print("Milvus 연결 완료: alias='sk_doc_test', db_name='sk_doc_test'")

# 임베딩 모델 초기화 (OpenAIEmbeddings 사용)
embedding_model = OpenAIEmbeddings(model='text-embedding-3-small')
app = FastAPI()


# ----------------------------------------
# PDF 텍스트 추출 및 전처리 함수
# ----------------------------------------
def extract_and_clean_pdf(file_obj, file_number: int) -> str:
    """
    주어진 PDF 파일 객체에서 텍스트를 추출하고,
    파일 번호에 따라 불필요한 문구를 제거한 후 각 페이지에 [PAGE_x] 마커를 추가한 텍스트를 반환합니다.

    Args:
        file_obj: PDF 파일 객체.
        file_number: 문서 유형을 구분하기 위한 번호.

    Returns:
        전체 페이지의 텍스트를 개행 문자로 연결한 문자열.
    """
    text_list = []
    try:
        with pdfplumber.open(file_obj) as pdf:
            for idx, page in enumerate(pdf.pages):
                raw_text = page.extract_text() or ""
                cleaned = clean_text(raw_text, file_number)
                # 각 페이지 앞에 [PAGE_x] 마커 추가 (페이지 번호는 1부터 시작)
                page_marker = f"[PAGE_{idx+1}]\n{cleaned}"
                text_list.append(page_marker)
        return "\n".join(text_list)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF 추출 오류: {e}")


def clean_text(text: str, file_number: int) -> str:
    """
    문서 번호에 따라 불필요한 문자열을 제거합니다.

    Args:
        text: 원본 텍스트.
        file_number: 문서 유형 구분 번호.

    Returns:
        정제된 텍스트.
    """
    if file_number == 2:
        text = re.sub(r"Ramboll - HAI LONG OFFSHORE WIND FARM", "", text)
        text = re.sub(r"HL-BOP-RAM-DES-SPC-00002, Rev\.4, 21/07/2022", "", text)
        text = re.sub(r"Uncontrolled copy when printed\. Most recent version is available on Aconex\.", "", text)
    elif file_number == 1:
        text = re.sub(r"Ramboll – FENGMIAO 1 OFFSHORE WIND FARM", "", text)
        text = re.sub(r"DocuSign Envelope ID: [A-Za-z0-9-]+", "", text)
        text = re.sub(r"Confidential", "", text)
        text = re.sub(r"Doc ID REN2021N00979-RAM-SP-00201 / Ramboll Version 1\.0 / Client Version 0", "", text)
    return text


# ----------------------------------------
# [PAGE_X] 마커 파싱 함수
# ----------------------------------------
def parse_page_marker_and_strip(chunk_text: str):
    """
    텍스트에서 [PAGE_x] 마커를 찾아, 최소 페이지 번호에서 1을 빼서 실제 PDF 페이지 번호와 맞춘 후,
    마커를 제거한 텍스트를 반환합니다.

    Args:
        chunk_text: [PAGE_x] 마커가 포함된 텍스트.

    Returns:
        tuple: (page_num, cleaned_text)
    """
    pattern = re.compile(r"\[PAGE_(\d+)\]")
    pages_found = pattern.findall(chunk_text)
    if pages_found:
        # 최소 페이지 번호에서 1을 빼서 실제 PDF 페이지 번호와 맞춤 (0부터 시작)
        min_page = min(int(p) for p in pages_found) - 1
    else:
        min_page = None
    cleaned_text = pattern.sub("", chunk_text).strip()
    return min_page, cleaned_text


# ----------------------------------------
# 목차 기반 분할 및 문단 분할 함수들
# ----------------------------------------
def extract_toc_headings(text, toc_marker_list=["목차", "Table of Contents", "Contents"], toc_line_limit=200):
    """
    텍스트에서 목차 영역을 찾아 목차 항목(heading)들을 추출합니다.

    Args:
        text: 전체 텍스트.
        toc_marker_list: 목차를 찾기 위한 키워드 리스트.
        toc_line_limit: 목차 추출 시 고려할 최대 라인 수.

    Returns:
        추출된 목차 항목 리스트.
    """
    if isinstance(text, list):
        text = "\n".join(text)
    lines = text.splitlines()
    toc_start = next((i for i, line in enumerate(lines[:500]) if any(marker in line for marker in toc_marker_list)), None)
    if toc_start is None:
        return []
    toc_block = "\n".join(lines[toc_start:toc_start + toc_line_limit])
    pattern = re.compile(r'^\s*(\d+(?:\.\d+)+\s+[^\n]+)', re.MULTILINE)
    headings = pattern.findall(toc_block)
    cleaned_headings = [re.sub(r'\s+\d+$', '', h).strip() for h in headings if h.strip()]
    return list(dict.fromkeys(cleaned_headings))


def split_text_by_heading_using_toc(text):
    """
    목차 항목을 기준으로 텍스트를 분할하여 각 섹션의 heading과 내용을 튜플로 반환합니다.

    Args:
        text: 전체 텍스트.

    Returns:
        리스트 of 튜플: [(heading, content), ...]
    """
    if isinstance(text, list):
        text = " ".join(text)
    toc_headings = extract_toc_headings(text)
    if not toc_headings:
        pattern = re.compile(r'(\d+(?:\.\d+)+\s+[^\n]+)')
        parts = pattern.split(text)
        sections = []
        for i in range(1, len(parts), 2):
            heading = parts[i].strip()
            content = parts[i+1].strip() if (i+1) < len(parts) else ""
            sections.append((heading, content))
        return sections
    else:
        pattern = '|'.join([re.escape(h) for h in toc_headings])
        parts = re.split(f'({pattern})', text)
        sections = []
        for i in range(1, len(parts), 2):
            heading = parts[i].strip()
            content = parts[i+1].strip() if (i+1) < len(parts) else ""
            sections.append((heading, content))
        return sections


def analyze_and_split_sections(paragraphs, label_key, content_key, max_length=700):
    """
    긴 문단을 지정한 최대 길이(max_length) 이하로 쪼개고, 
    원래 문단의 label 및 page_num 정보를 유지한 채로 새 청크 리스트를 반환합니다.

    Args:
        paragraphs: 문단 리스트 (각 문단은 딕셔너리, label, content, page_num 포함).
        label_key: 문단 label의 키 (예: "label").
        content_key: 문단 내용의 키 (예: "content").
        max_length: 최대 문단 길이 제한.

    Returns:
        새로 분할된 문단 리스트.
    """
    def split_into_segments(content):
        raw_segments = [seg.strip() for seg in content.split("\n\n") if seg.strip()]
        merged_segments = []
        current_segment = ""
        for seg in raw_segments:
            # 길이가 짧은 문장은 병합
            if len(seg) < 200:
                current_segment += ("\n\n" + seg) if current_segment else seg
            else:
                if current_segment:
                    merged_segments.append(current_segment)
                    current_segment = ""
                merged_segments.append(seg)
        if current_segment:
            merged_segments.append(current_segment)
        final_segments = []
        for segment in merged_segments:
            if len(segment) > max_length:
                final_segments.extend(split_long_segment(segment, max_length=700, min_chunk_length=300))
            else:
                final_segments.append(segment)
        return final_segments

    def split_long_segment(segment, max_length=700, min_chunk_length=300, min_chunk_size=50):
        chunks = []
        text = segment
        sentence_end_pattern = re.compile(r"(?<=[A-Za-z])\.(?=\n)")
        fallback_pattern = re.compile(r"\.")
        while len(text) > max_length:
            split_point = -1
            for match in sentence_end_pattern.finditer(text[min_chunk_length:max_length + 1]):
                split_point = match.end() + min_chunk_length - 1
                break
            if split_point == -1:
                for match in fallback_pattern.finditer(text[max_length + 1:]):
                    split_point = match.end() + max_length
                    break
            if split_point == -1:
                split_point = max_length
            chunk = text[:split_point].strip()
            chunks.append(chunk)
            text = text[split_point:].strip()
        if text:
            chunks.append(text.strip())
        return chunks

    new_paragraphs = []
    for paragraph in paragraphs:
        content = paragraph[content_key]
        page_num = paragraph.get("page_num", None)
        segments = split_into_segments(content)
        if len(segments) > 1:
            for idx, seg in enumerate(segments):
                new_paragraphs.append({
                    label_key: f"{paragraph[label_key]} - Part {idx+1}",
                    content_key: seg,
                    "page_num": page_num
                })
        else:
            paragraph["content"] = content
            new_paragraphs.append(paragraph)
    return new_paragraphs


# --------------------------------------------------
# MilvusPDFProcessor 클래스: Milvus DB에 파일 저장 및 검색
# --------------------------------------------------
class MilvusPDFProcessor:
    def __init__(self, collection_name="sk_doc_test13", alias="sk_doc_test"):
        """
        MilvusPDFProcessor 클래스 생성자.
        지정된 alias와 db_name("sk_doc_test")을 사용하여 Milvus에 연결하고,
        스키마에 따라 컬렉션을 생성하거나 기존 컬렉션에 연결합니다.

        Args:
            collection_name: 사용할 컬렉션 이름.
            alias: Milvus 연결 alias.
        """
        self.alias = alias
        self.uri = "49.50.175.45"
        self.port = "19530"
        self.collection_name = collection_name

        # Milvus에 연결 (이미 최상단에서 연결되었지만, alias를 재확인)
        connections.connect(alias=self.alias, host=self.uri, port=self.port, db_name="sk_doc_test")

        # 스키마 정의 (file_id는 VARCHAR로 저장)
        self.fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="file_id", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="heading", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),
            FieldSchema(name="metadatas", dtype=DataType.JSON),
        ]
        self.schema = CollectionSchema(fields=self.fields, description="SK document collection")

        # 컬렉션 존재 여부 확인 후 생성 또는 연결
        if not utility.has_collection(self.collection_name, using=self.alias):
            print(f"'{self.collection_name}' 컬렉션이 존재하지 않습니다. 새로 생성합니다.")
            self.collection = Collection(name=self.collection_name, schema=self.schema, using=self.alias)
            print(f"'{self.collection_name}' 컬렉션이 성공적으로 생성되었습니다.")
        else:
            print(f"'{self.collection_name}' 컬렉션이 이미 존재합니다. 기존 컬렉션에 연결합니다.")
            self.collection = Collection(name=self.collection_name, using=self.alias)

        # 인덱스 생성
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 128}
        }
        if not self.collection.has_index():
            self.collection.create_index(field_name="embedding", index_params=index_params)
            print(f"Index created successfully on '{self.collection_name}'!")
        else:
            print(f"Index already exists on '{self.collection_name}'.")

    def insert_data(self, data):
        """
        Milvus 컬렉션에 데이터를 삽입합니다.

        Args:
            data: 삽입할 데이터 (리스트 형태)
        """
        self.collection.insert(data)
        print("데이터가 성공적으로 삽입되었습니다.")

    def is_file_stored(self, file_id: str) -> bool:
        """
        주어진 file_id가 컬렉션에 이미 저장되어 있는지 확인합니다.

        Args:
            file_id: 파일 고유 식별자.

        Returns:
            bool: 저장되어 있으면 True, 아니면 False.
        """
        self.collection.load()
        expr = f'file_id == "{file_id}"'
        res = self.collection.query(expr=expr, output_fields=["file_id"])
        return len(res) > 0

    def save_vector_db(self, file_id: str, processed_chunks, embeddings_model):
        """
        file_id에 해당하는 파일이 아직 저장되지 않았다면,
        처리된 청크 데이터를 임베딩 생성 후 Milvus 컬렉션에 저장합니다.

        Args:
            file_id: 파일 고유 식별자.
            processed_chunks: 분할된 문단 리스트.
            embeddings_model: 임베딩 생성 모델.
        """
        if self.is_file_stored(file_id):
            print(f"이미 file_id={file_id}로 저장된 문서이므로, 중복 저장 스킵합니다.")
            return
        for chunk in processed_chunks:
            heading = chunk["label"]
            content = chunk["content"].strip() if chunk["content"] else ""
            if not content:
                continue
            page_num = chunk.get("page_num", None)
            page_str = "" if page_num is None else str(page_num)
            metadata = {
                "file_id": file_id,
                "heading": heading,
                "content": content,
                "page": page_str
            }
            vector_embedding = embeddings_model.embed_query(content)
            data = [
                [file_id],
                [heading],
                [content],
                [vector_embedding],
                [metadata]
            ]
            self.insert_data(data)

    def query_vector_db(self, query_text, embeddings_model, top_k=5, file_id_filter=None):
        """
        주어진 텍스트에 대한 임베딩을 생성하고, file_id_filter가 지정되면
        해당 file_id에 해당하는 문서만 검색하여 상위 top_k 결과를 반환합니다.

        Args:
            query_text: 검색할 텍스트.
            embeddings_model: 임베딩 생성 모델.
            top_k: 반환할 상위 결과 개수.
            file_id_filter: 검색할 파일의 file_id (문자열).

        Returns:
            각 결과 항목: (heading, content, page, score)
        """
        self.collection.load()
        query_embedding = embeddings_model.embed_query(query_text)
        expr = None
        if file_id_filter is not None:
            expr = f'file_id == "{file_id_filter}"'
        search_results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=top_k,
            expr=expr,
            output_fields=["file_id", "heading", "content", "metadatas"]
        )
        final_results = []
        for result in search_results:
            for hit in result:
                hit_heading = getattr(hit.entity, "heading", "")
                hit_content = getattr(hit.entity, "content", "")
                metadata = getattr(hit.entity, "metadatas", {})
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except Exception:
                        metadata = {}
                hit_page = ""
                if isinstance(metadata, dict):
                    hit_page = metadata.get("page", "")
                elif isinstance(metadata, list) and len(metadata) > 0:
                    hit_page = metadata[0].get("page", "")
                score = hit.distance
                final_results.append((hit_heading, hit_content, hit_page, score))
        return final_results

    def highlight_differences(self, content1, content2):
        """
        두 텍스트를 줄 단위로 비교하여, 추가된 줄과 삭제된 줄을 반환합니다.

        Args:
            content1: 기준 텍스트.
            content2: 비교 텍스트.

        Returns:
            tuple: (added_content, deleted_content)
        """
        lines1 = set(content1.splitlines()) if content1 else set()
        lines2 = set(content2.splitlines()) if content2 else set()
        added = lines1 - lines2
        deleted = lines2 - lines1
        added_content = "\n".join(f"Added: {line}" for line in added) if added else ""
        deleted_content = "\n".join(f"Deleted: {line}" for line in deleted) if deleted else ""
        return added_content, deleted_content

    def process_chunk(self, chunk, idx, milvus_db_2, file_id_filter=None):
        """
        단일 문단 청크를 처리하여, 기준 텍스트와 비교 후 가장 유사한 결과를 반환합니다.

        Args:
            chunk: 처리할 문단 청크 (딕셔너리, label, content, page_num 포함).
            idx: 청크 인덱스.
            milvus_db_2: 검색에 사용할 MilvusPDFProcessor 객체.
            file_id_filter: 검색할 파일의 file_id (문자열).

        Returns:
            딕셔너리: {'Page_1', 'Heading_1', 'Content_1', 'Page_2', 'Heading_2', 'Content_2', 'Added_Content', 'Deleted_Content', 'Content_similarity'}
        """
        content_1 = chunk['content'].strip() if chunk['content'] else ""
        label_1 = chunk['label']
        page_num_1 = chunk.get("page_num", None)
        page1 = "" if page_num_1 is None else str(page_num_1)
        if not content_1:
            return {
                'Page_1': page1,
                'Heading_1': label_1,
                'Content_1': content_1,
                'Page_2': "",
                'Heading_2': "",
                'Content_2': "",
                'Added_Content': "",
                'Deleted_Content': "",
                'Content_similarity': ""
            }
        similar_chunks = milvus_db_2.query_vector_db(content_1, embedding_model, top_k=5, file_id_filter=file_id_filter)
        if similar_chunks:
            best_heading, best_content, best_page, best_score = similar_chunks[0]
            content_similarity = best_score
            if not best_page:
                best_page = ""
        else:
            best_heading = ""
            best_content = ""
            best_page = ""
            content_similarity = ""
        added_diff, deleted_diff = self.highlight_differences(content_1, best_content)
        return {
            'Page_1': page1,
            'Heading_1': label_1,
            'Content_1': content_1,
            'Page_2': best_page,
            'Heading_2': best_heading,
            'Content_2': best_content,
            'Added_Content': added_diff,
            'Deleted_Content': deleted_diff,
            'Content_similarity': content_similarity
        }

    def process_chunks_in_parallel(self, chunks, milvus_db_2, file_id_filter=None):
        """
        여러 문단 청크를 병렬로 처리하여 각 청크의 결과를 리스트로 반환합니다.

        Args:
            chunks: 문단 청크 리스트.
            milvus_db_2: 검색에 사용할 MilvusPDFProcessor 객체.
            file_id_filter: 검색할 파일의 file_id (문자열).

        Returns:
            처리된 청크 결과 리스트.
        """
        results = [None] * len(chunks)
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_idx = {
                executor.submit(self.process_chunk, chunk, idx, milvus_db_2, file_id_filter=file_id_filter): idx
                for idx, chunk in enumerate(chunks)
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
        return results

    def get_file_headings(self, file_id: str):
        """
        주어진 file_id에 해당하는 모든 문단 청크에서 heading만 조회하여 리스트로 반환합니다.

        Args:
            file_id: 파일 고유 식별자 (문자열).

        Returns:
            heading 리스트.
        """
        self.collection.load()
        expr = f'file_id == "{file_id}"'
        results = self.collection.query(expr=expr, output_fields=["heading"])
        headings = [r["heading"] for r in results if "heading" in r]
        return headings


# -----------------------------------------------------
# /compare 엔드포인트: PDF 비교 및 Excel 결과 반환
# -----------------------------------------------------
@app.post("/compare")
async def compare_pdfs(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    """
    두 PDF 파일을 받아 텍스트를 추출, 목차 기반 분할 및 긴 문단 분할 후 Milvus DB에 저장합니다.
    이후, 기준문서(file1)의 청크와 비교문서(file2)의 데이터를 비교하여 결과를 Excel 파일로 반환합니다.

    Args:
        file1: 기준 문서 PDF 파일.
        file2: 비교 문서 PDF 파일.

    Returns:
        Excel 파일 응답 (comparison_result.xlsx).
    """
    file1_id = file1.filename
    file2_id = file2.filename

    text1 = extract_and_clean_pdf(file1.file, file_number=1)
    text2 = extract_and_clean_pdf(file2.file, file_number=2)

    sections1 = split_text_by_heading_using_toc(text1)
    sections2 = split_text_by_heading_using_toc(text2)

    paragraphs1 = [{"label": h, "content": c} for h, c in sections1]
    paragraphs2 = [{"label": h, "content": c} for h, c in sections2]

    for p in paragraphs1:
        pnum, ctext = parse_page_marker_and_strip(p["content"])
        p["page_num"] = pnum
        p["content"] = ctext
    for p in paragraphs2:
        pnum, ctext = parse_page_marker_and_strip(p["content"])
        p["page_num"] = pnum
        p["content"] = ctext

    paragraphs1 = analyze_and_split_sections(paragraphs1, "label", "content")
    paragraphs2 = analyze_and_split_sections(paragraphs2, "label", "content")

    processor = MilvusPDFProcessor(collection_name="sk_doc_test13", alias="sk_doc_test")

    for chunk in paragraphs1:
        chunk["metadata"] = {"file_id": file1_id}
    processor.save_vector_db(file1_id, paragraphs1, embedding_model)

    for chunk in paragraphs2:
        chunk["metadata"] = {"file_id": file2_id}
    processor.save_vector_db(file2_id, paragraphs2, embedding_model)

    results = processor.process_chunks_in_parallel(paragraphs1, processor, file_id_filter=file2_id)

    df = pd.DataFrame(results, columns=[
        'Page_1', 'Heading_1', 'Content_1',
        'Page_2', 'Heading_2', 'Content_2',
        'Added_Content', 'Deleted_Content', 'Content_similarity'
    ])

    def download_excel(df):
        output = io.BytesIO()
        df.to_excel(output, index=False, engine="openpyxl")
        output.seek(0)
        file_path = "comparison_result.xlsx"
        with open(file_path, 'wb') as f:
            f.write(output.read())
        headers = {'Content-Disposition': 'attachment; filename="comparison_result.xlsx"'}
        return FileResponse(
            path=file_path,
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            headers=headers
        )

    return download_excel(df)


# -----------------------------------------------------
# /get_headings 엔드포인트: 문서 헤딩 조회
# -----------------------------------------------------
@app.get("/get_headings")
async def get_headings(file1_id: str, file2_id: str):
    """
    주어진 file1_id(기준문서)와 file2_id(비교문서)에 해당하는 Milvus DB의 모든 heading을 조회하여,
    헤딩 총 개수를 콘솔에 출력한 후 JSON 형태로 반환합니다.

    Query Parameters:
        file1_id: 기준 문서의 file_id.
        file2_id: 비교 문서의 file_id.

    Returns:
        JSON 객체: {"file1_headings": [...], "file2_headings": [...]}
    """
    processor = MilvusPDFProcessor(collection_name="sk_doc_test13", alias="sk_doc_test")
    file1_headings = processor.get_file_headings(file1_id)
    file2_headings = processor.get_file_headings(file2_id)
    # 콘솔에 출력 (헤딩 총 개수와 헤딩 리스트)
    print(f"file1_headings({len(file1_headings)}) :", file1_headings)
    print(f"file2_headings({len(file2_headings)}) :", file2_headings)
    return {
        "file1_headings": file1_headings,
        "file2_headings": file2_headings
    }
