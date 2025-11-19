"""
Scientific Paper OCR Processor using DeepSeek OCR model.
This module provides functionality to convert PDF papers into machine-readable text
while preserving the document structure in markdown format.
"""

from typing import List, Optional
from transformers import AutoModel, AutoTokenizer
import torch
import os
import pdf2image  # type: ignore
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScientificPaperOCR: 
    MODEL_NAME = "deepseek-ai/DeepSeek-OCR"
    PROMPT = "<image>\n<|grounding|>Convert the document to markdown."
    
    def __init__(
        self,
        gpu_device: str = '0',
        model_name: str = MODEL_NAME,
        base_size: int = 1280,
        image_size: int = 640,
        crop_mode: bool = False
    ):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device
        self.model_name = model_name
        self.base_size = base_size
        self.image_size = image_size
        self.crop_mode = crop_mode
        self.img_paths: List[str] = []
        
        logger.info(f"Initializing ScientificPaperOCR with model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            use_safetensors=True,
        )
        logger.info("Model and tokenizer initialized successfully")

    def pdf_to_images(self, pdf_path: str, dpi: int = 300, output_dir: str = "") -> List[str]:
        logger.info(f"Converting PDF to images: {pdf_path}")
        self.img_paths = []
        
        pages = pdf2image.convert_from_path(pdf_path, dpi=dpi)
        for i, page in enumerate(pages, 1):
            img = page.convert("RGB")
            output_path = os.path.join(output_dir, f"page_{i}.jpg")
            self.img_paths.append(output_path)
            img.save(output_path, "JPEG")
            
        logger.info(f"Converted {len(pages)} pages to images")
        return self.img_paths

    def process_image(self, image_file: str, output_dir: str) -> None:
        logger.info(f"Processing image: {image_file}")
        
        self.model.infer(
            self.tokenizer,
            prompt=self.PROMPT,
            image_file=image_file,
            output_path=output_dir,
            base_size=self.base_size,
            image_size=self.image_size,
            crop_mode=self.crop_mode,
            save_results=True,
            test_compress=False,
        )

    def process_pdf(
        self,
        pdf_path: str,
        output_dir: str,
        text_output_path: Optional[str] = None,
        dpi: int = 300
    ) -> str:
        logger.info(f"Starting OCR pipeline for: {pdf_path}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.pdf_to_images(pdf_path=pdf_path, dpi=dpi, output_dir=output_dir)
        
        for i, img_file in enumerate(self.img_paths, 1):
            output_ocr = os.path.join(output_dir, f"page_{i}")
            os.makedirs(output_ocr, exist_ok=True)
            self.process_image(image_file=img_file, output_dir=output_ocr)
            
        if text_output_path is None:
            text_output_path = os.path.join(output_dir, "final_text.txt")
        
        self.combine_results(ocr_dir=output_dir, text_path=text_output_path)
        return text_output_path

    def combine_results(self, ocr_dir: str, text_path: str) -> None:
        logger.info(f"Combining results into: {text_path}")
        
        os.makedirs(os.path.dirname(text_path), exist_ok=True)
        with open(text_path, "w") as text_file:
            for i in range(len(self.img_paths)):
                mmd_path = os.path.join(ocr_dir, f"page_{i+1}/result.mmd")
                with open(mmd_path, "r") as mmd_file:
                    mmd = mmd_file.read()
                text_file.write(mmd)
                text_file.write("\n")
                
        logger.info("Results combined successfully")

def main():
    ocr_processor = ScientificPaperOCR()
    
    pdf_path = "../../data/raw/NIPS-2017-attention-is-all-you-need-Paper.pdf"
    output_dir = "../../data/ocr"
    text_output = "../../data/processed/final_text.txt"
    
    ocr_processor.process_pdf(
        pdf_path=pdf_path,
        output_dir=output_dir,
        text_output_path=text_output,
        dpi=400
    )

if __name__ == "__main__":
    main()