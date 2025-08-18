# integration_bridge.py
# Integration bridge connecting your existing brand analysis system with FastAPI

import asyncio
import time
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import logging

# Import your existing classes from app5.py
from app5 import (
    EnhancedDocumentProcessor,
    DocumentChunker,
    BrandAnalysisOrchestrator,
    BatchProcessingOrchestrator,
    EnhancedBrandAnalysisOrchestrator,
    ObjectiveDataBedrock,
    SynthesisReport,
    AggregatedExtractionResult
)

logger = logging.getLogger(__name__)

@dataclass
class BrandAnalysisConfig:
    """Configuration for brand analysis processing"""
    use_batch_processing: Optional[bool] = None  # Auto-detect if None
    chunk_size: int = 8000
    overlap: int = 800
    max_concurrent_chunks: int = 3
    temperature: float = 0.1
    max_tokens: int = 4000

@dataclass
class ProcessingResult:
    """Results from brand analysis processing"""
    processing_method: str  # "batch" or "traditional"
    total_files: int
    total_chunks: Optional[int]
    total_words: int
    total_characters: int
    processing_duration: float
    chunks_processed: Optional[int]
    objective_data: Dict[str, Any]
    synthesis_report: Dict[str, Any]
    agent_results: Dict[str, Any]
    model_settings: Dict[str, Any]

class FastAPIIntegrationBridge:
    """
    Bridge class that adapts your existing brand analysis system for FastAPI
    """
    
    def __init__(self):
        self.doc_processor = EnhancedDocumentProcessor()
        self.enhanced_orchestrator = EnhancedBrandAnalysisOrchestrator()
        
    def _read_file_content(self, file_path: str) -> str:
        """Read content from a file path"""
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return ""
    
    def _create_mock_uploaded_files(self, file_paths: List[str]) -> List:
        """
        Create mock uploaded file objects that mimic Streamlit's UploadedFile interface
        """
        class MockUploadedFile:
            def __init__(self, file_path: str):
                self.file_path = Path(file_path)
                self.name = self.file_path.name
                self.type = self._get_mime_type()
                self._content = None
                self._position = 0
                
            def _get_mime_type(self):
                suffix = self.file_path.suffix.lower()
                if suffix == '.md':
                    return 'text/markdown'
                elif suffix == '.txt':
                    return 'text/plain'
                elif suffix == '.docx':
                    return 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
                else:
                    return 'text/plain'
            
            def read(self):
                if self._content is None:
                    with open(self.file_path, 'rb') as f:
                        self._content = f.read()
                
                if self._position == 0:
                    self._position = len(self._content)
                    return self._content
                else:
                    return b''  # Simulate end of file
            
            def seek(self, position: int):
                self._position = position
        
        return [MockUploadedFile(path) for path in file_paths]
    
    async def process_files_and_create_batch_data(
        self, 
        file_paths: List[str], 
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Process files and create batch data using your existing logic
        """
        try:
            if progress_callback:
                progress_callback(2, 35, "Creating batch data from uploaded files")
            
            # Create mock uploaded files
            mock_files = self._create_mock_uploaded_files(file_paths)
            
            # Use your existing document processor
            batch_data = self.doc_processor.process_multiple_files_with_chunking(mock_files)
            
            if progress_callback:
                progress_callback(2, 50, "Batch data created successfully", {
                    "total_files": batch_data["total_files"],
                    "total_chunks": batch_data["total_chunks"],
                    "total_characters": batch_data["total_characters"]
                })
            
            return batch_data
            
        except Exception as e:
            logger.error(f"Error creating batch data: {e}")
            raise
    
    async def run_brand_analysis(
        self,
        batch_data: Dict[str, Any],
        sitemap_content: str,
        config: BrandAnalysisConfig,
        progress_callback: Optional[Callable] = None
    ) -> ProcessingResult:
        """
        Run brand analysis using your existing orchestrator system
        """
        start_time = time.time()
        
        try:
            # Determine processing method
            use_batch = config.use_batch_processing
            if use_batch is None:
                # Auto-detect based on your existing logic
                use_batch = self.enhanced_orchestrator.should_use_batch_processing(batch_data)
            
            processing_method = "batch" if use_batch else "traditional"
            
            if progress_callback:
                progress_callback(3, 55, f"Starting {processing_method} analysis", {
                    "processing_method": processing_method,
                    "use_batch_processing": use_batch
                })
            
            if use_batch:
                # Use batch processing
                if progress_callback:
                    progress_callback(3, 60, "Running batch analysis with smart chunking")
                
                aggregated_results, synthesis_report, agent_results = await self.enhanced_orchestrator.batch_processor.run_batch_analysis(
                    batch_data, sitemap_content
                )
                
                if progress_callback:
                    progress_callback(4, 80, "Batch analysis completed, synthesizing results")
                
                # Convert aggregated results to standard format
                objective_data = self._convert_aggregated_to_objective(aggregated_results)
                
            else:
                # Use traditional processing
                if progress_callback:
                    progress_callback(3, 60, "Running traditional analysis")
                
                # Convert batch data to traditional format
                combined_content = self.enhanced_orchestrator._batch_to_traditional_format(batch_data)
                
                # Use traditional orchestrator
                traditional_orchestrator = BrandAnalysisOrchestrator()
                objective_data, synthesis_report, agent_results = await traditional_orchestrator.run_analysis(
                    combined_content, sitemap_content
                )
                
                if progress_callback:
                    progress_callback(4, 80, "Traditional analysis completed")
            
            processing_duration = time.time() - start_time
            
            if progress_callback:
                progress_callback(4, 90, "Preparing final results")
            
            # Create processing result
            result = ProcessingResult(
                processing_method=processing_method,
                total_files=batch_data["total_files"],
                total_chunks=batch_data.get("total_chunks"),
                total_words=batch_data["total_words"],
                total_characters=batch_data["total_characters"],
                processing_duration=processing_duration,
                chunks_processed=agent_results.get("total_chunks_processed") if use_batch else None,
                objective_data=self._serialize_objective_data(objective_data),
                synthesis_report=self._serialize_synthesis_report(synthesis_report),
                agent_results=agent_results,
                model_settings={
                    "temperature": config.temperature,
                    "max_tokens": config.max_tokens,
                    "chunk_size": config.chunk_size,
                    "overlap": config.overlap
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Brand analysis error: {e}")
            raise
    
    def _convert_aggregated_to_objective(self, aggregated_results: AggregatedExtractionResult) -> ObjectiveDataBedrock:
        """Convert aggregated results to ObjectiveDataBedrock format"""
        try:
            from app5 import (
                AuditMetadata, CorpusBaseline, NarrativeDataPoints,
                VerbalIdentityDataPoints, AudienceData, EmergentBusinessStructures,
                NuancedSignals, ObjectiveDataBedrock
            )
            
            return ObjectiveDataBedrock(
                audit_metadata=AuditMetadata(
                    audit_date=datetime.now().strftime('%Y-%m-%d'),
                    agent_name="The Auditor (Batch Processed)",
                    persona="Batch Processing Data Extractor"
                ),
                corpus_baseline=aggregated_results.corpus_baseline,
                narrative_data_points=aggregated_results.narrative_data_points,
                verbal_identity_data_points=aggregated_results.verbal_identity_data_points,
                audience_data=aggregated_results.audience_data,
                emergent_business_structures=aggregated_results.emergent_business_structures,
                relationship_matrix=aggregated_results.relationship_matrix,
                nuanced_signals=aggregated_results.nuanced_signals
            )
        except Exception as e:
            logger.error(f"Error converting aggregated results: {e}")
            raise
    
    def _serialize_objective_data(self, objective_data: ObjectiveDataBedrock) -> Dict[str, Any]:
        """Serialize ObjectiveDataBedrock to dictionary"""
        try:
            if hasattr(objective_data, 'model_dump'):
                return objective_data.model_dump()
            elif hasattr(objective_data, 'dict'):
                return objective_data.dict()
            else:
                # Fallback manual serialization
                return {
                    "audit_metadata": {
                        "audit_date": objective_data.audit_metadata.audit_date,
                        "agent_name": objective_data.audit_metadata.agent_name,
                        "persona": objective_data.audit_metadata.persona
                    },
                    "corpus_baseline": {
                        "total_pages_analyzed": objective_data.corpus_baseline.total_pages_analyzed,
                        "total_words_analyzed": objective_data.corpus_baseline.total_words_analyzed,
                        "total_pages_excluded": objective_data.corpus_baseline.total_pages_excluded
                    },
                    # Add other fields as needed
                }
        except Exception as e:
            logger.error(f"Error serializing objective data: {e}")
            return {}
    
    def _serialize_synthesis_report(self, synthesis_report: SynthesisReport) -> Dict[str, Any]:
        """Serialize SynthesisReport to dictionary"""
        try:
            if hasattr(synthesis_report, 'model_dump'):
                return synthesis_report.model_dump()
            elif hasattr(synthesis_report, 'dict'):
                return synthesis_report.dict()
            else:
                # Fallback manual serialization
                return {
                    "report_markdown": synthesis_report.report_markdown,
                    "analysis_date": synthesis_report.analysis_date,
                    "summary": synthesis_report.summary
                }
        except Exception as e:
            logger.error(f"Error serializing synthesis report: {e}")
            return {}

# Global bridge instance
integration_bridge = FastAPIIntegrationBridge()

async def process_files_and_analyze(
    file_paths: List[str],
    sitemap_path: Optional[str] = None,
    config: Optional[BrandAnalysisConfig] = None,
    progress_callback: Optional[Callable] = None
) -> ProcessingResult:
    """
    Main entry point for processing files and running brand analysis
    
    Args:
        file_paths: List of file paths to analyze
        sitemap_path: Optional path to sitemap file
        config: Analysis configuration
        progress_callback: Function to call with progress updates
    
    Returns:
        ProcessingResult with complete analysis results
    """
    if config is None:
        config = BrandAnalysisConfig()
    
    try:
        # Step 1: Validate files
        if progress_callback:
            progress_callback(1, 10, "Validating input files")
        
        for file_path in file_paths:
            if not Path(file_path).exists():
                raise FileNotFoundError(f"File not found: {file_path}")
        
        if progress_callback:
            progress_callback(1, 20, "Files validated successfully")
        
        # Step 2: Process files and create batch data
        batch_data = await integration_bridge.process_files_and_create_batch_data(
            file_paths, progress_callback
        )
        
        # Step 3: Read sitemap content
        sitemap_content = ""
        if sitemap_path and Path(sitemap_path).exists():
            if progress_callback:
                progress_callback(2, 52, "Reading sitemap file")
            sitemap_content = integration_bridge._read_file_content(sitemap_path)
        
        if progress_callback:
            progress_callback(2, 55, "File processing completed")
        
        # Step 4: Run brand analysis
        result = await integration_bridge.run_brand_analysis(
            batch_data, sitemap_content, config, progress_callback
        )
        
        if progress_callback:
            progress_callback(5, 100, "Analysis completed successfully")
        
        return result
        
    except Exception as e:
        logger.error(f"Process files and analyze error: {e}")
        if progress_callback:
            progress_callback(1, 0, f"Error: {str(e)}")
        raise

# Utility functions for FastAPI integration

def validate_file_paths(file_paths: List[str]) -> bool:
    """Validate that all file paths exist and are readable"""
    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            return False
        if not path.is_file():
            return False
        try:
            with open(path, 'r', encoding='utf-8') as f:
                f.read(1)  # Try to read first character
        except Exception:
            return False
    return True

def estimate_processing_time(total_size_bytes: int, total_files: int) -> int:
    """Estimate processing time in seconds based on file size and count"""
    # Base time: 30 seconds
    base_time = 30
    
    # Size factor: 1 second per 100KB
    size_factor = total_size_bytes / (100 * 1024)
    
    # File factor: 10 seconds per file
    file_factor = total_files * 10
    
    # Complexity factor for multiple files
    complexity_factor = min(total_files * 5, 60)
    
    estimated_seconds = base_time + size_factor + file_factor + complexity_factor
    
    # Cap between 1 minute and 30 minutes
    return max(60, min(int(estimated_seconds), 1800))

def get_processing_method_recommendation(total_size: int, total_files: int) -> str:
    """Recommend processing method based on file characteristics"""
    if total_size > 50000 or total_files > 3:
        return "batch"
    else:
        return "traditional"

# Export main functions for FastAPI
__all__ = [
    'process_files_and_analyze',
    'BrandAnalysisConfig',
    'ProcessingResult',
    'validate_file_paths',
    'estimate_processing_time',
    'get_processing_method_recommendation'
]
