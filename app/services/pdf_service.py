import io
from pypdf import PdfReader
from fastapi import HTTPException,  status

def extract_pages(pdf_bytes:bytes) -> list[str]:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not read the pdf."
        )
    
    pages=[]
    for page in reader.pages:
        text=page.extract_text() or ""
        pages.append(text.strip())

    non_empty = [p for p in pages if p]

    if not non_empty:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No readable text found."
        )
    return non_empty



def build_chunks(pages:list[str])->list[str]:
    pages_per_chunk=3
    overlap=1

    chunks=[]
    step = pages_per_chunk-overlap
    
    for start in range(0,len(pages),step):
        window = pages[start:start+pages_per_chunk]
        if not window:
            break
        chunks.append("\n\n".join(window))

        if start+pages_per_chunk >=len(pages):
            break
    return chunks