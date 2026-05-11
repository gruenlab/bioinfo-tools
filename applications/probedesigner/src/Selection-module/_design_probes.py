"""
Complete Probe Design Pipeline - Hybrid Approach
=================================================
Combines reference preparation with ODT pipeline design.

Stage 1: Prepare genomic regions and FASTA files from genes
Stage 2: Run method-specific probe design pipeline (SCRINSHOT/MERFISH/seqFISH+)
"""

import os
from pathlib import Path
import pandas as pd
from typing import List, Dict, Optional

# oligo_designer_toolsuite is an optional dependency.
# When it is not installed the rest of the selection pipeline works normally;
# only the ODT probe-design stage will be unavailable.
try:
    from oligo_designer_toolsuite.database import OligoDatabase
    from oligo_designer_toolsuite.sequence_generator import (
        FtpLoaderEnsembl,
        FtpLoaderNCBI,
        CustomGenomicRegionGenerator,
    )
    from oligo_designer_toolsuite.pipelines import (
        ScrinshotProbeDesigner,
        MerfishProbeDesigner,
        SeqFishPlusProbeDesigner,
    )
    ODT_AVAILABLE = True
except ImportError:
    ODT_AVAILABLE = False
    OligoDatabase = None
    FtpLoaderEnsembl = FtpLoaderNCBI = CustomGenomicRegionGenerator = None
    ScrinshotProbeDesigner = MerfishProbeDesigner = SeqFishPlusProbeDesigner = None


def setup_directories(base_dir="./oligo_design_output"):
    """Create directory structure for outputs"""
    base_path = Path(base_dir)
    dirs = {
        'base': base_path,
        'references': base_path / 'references',
        'sequences': base_path / 'sequences',
        'results': base_path / 'results',
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs


def convert_gene_symbols_to_ids(gene_symbols: List[str], gtf_file: str, source: str = "ensembl") -> Dict[str, str]:
    """
    Convert gene symbols to gene IDs by parsing the GTF file.
    
    Parameters:
    -----------
    gene_symbols : list
        List of gene symbols (e.g., ['Gapdh', 'Actb'])
    gtf_file : str
        Path to GTF annotation file
    source : str
        'ensembl' or 'ncbi' - determines ID format to extract
    
    Returns:
    --------
    dict : Mapping of symbol -> gene_id (e.g., {'Gapdh': 'ENSMUSG00000057666'})
    """
    import re
    
    symbol_to_id = {}
    symbols_lower = {s.lower(): s for s in gene_symbols}  # Case-insensitive lookup
    
    print(f"\n  Converting {len(gene_symbols)} gene symbols to IDs...")
    
    with open(gtf_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            
            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue
            
            attributes = fields[8]
            
            # Extract gene_id
            if source == "ensembl":
                gene_id_match = re.search(r'gene_id "([^"]+)"', attributes)
            else:  # ncbi
                gene_id_match = re.search(r'gene_id "([^"]+)"', attributes)
            
            # Extract gene_name/symbol
            gene_name_match = re.search(r'gene_name "([^"]+)"', attributes)
            
            if gene_id_match and gene_name_match:
                gene_id = gene_id_match.group(1)
                gene_name = gene_name_match.group(1)
                
                # Check if this symbol is in our list (case-insensitive)
                if gene_name.lower() in symbols_lower:
                    original_symbol = symbols_lower[gene_name.lower()]
                    symbol_to_id[original_symbol] = gene_id
            
            # Stop early if we've found all symbols
            if len(symbol_to_id) == len(gene_symbols):
                break
    
    # Report conversion results
    found = list(symbol_to_id.keys())
    missing = [s for s in gene_symbols if s not in symbol_to_id]
    
    if found:
        print(f"  ✓ Converted {len(found)} symbols to IDs:")
        for symbol in found[:5]:  # Show first 5
            print(f"    {symbol} -> {symbol_to_id[symbol]}")
        if len(found) > 5:
            print(f"    ... and {len(found) - 5} more")
    
    if missing:
        print(f"  ✗ Could not find {len(missing)} symbols in GTF:")
        for symbol in missing[:10]:
            print(f"    {symbol}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")
    
    return symbol_to_id


class CompleteProbeDesignPipeline:
    """
    Complete pipeline combining genomic region extraction with probe design.

    Requires the optional dependency ``oligo-designer-toolsuite``.
    Install it with::

        pip install oligo-designer-toolsuite
        # or, inside the probe_design environment:
        uv pip install oligo-designer-toolsuite

    This class handles:
    1. Downloading genome/annotation references
    2. Extracting target gene sequences
    3. Running method-specific probe design (SCRINSHOT/MERFISH/seqFISH+)
    """

    def __init__(
        self,
        pipeline_type: str = "merfish",
        output_dir: str = "./oligo_design_output",
        species: str = "mus_musculus",
        annotation_source: str = "ensembl",
        annotation_release: str = "110",
        taxon: str = "vertebrate_mammalian",
        design_mode="filter_only",
        write_intermediate_steps: bool = True,
        n_jobs: int = 4
    ):
        """
        Initialize the complete probe design pipeline.
        
        Parameters:
        -----------
        pipeline_type : str
            'scrinshot', 'merfish', or 'seqfishplus'
        output_dir : str
            Base output directory
        species : str
            Species name (e.g., 'mus_musculus', 'homo_sapiens')
        annotation_source : str
            'ensembl' or 'ncbi'
        annotation_release : str
            Release version
        taxon : str
            For NCBI: 'vertebrate_mammalian', etc.
        design_mode : str
            'filter_only': Run only target probe design for gene filtering (fast)
            'complete': Run full pipeline including detection oligos/readout/primers (slow)
        write_intermediate_steps : bool
            Save intermediate files
        n_jobs : int
            Number of parallel jobs
        """
        if not ODT_AVAILABLE:
            raise ImportError(
                "oligo_designer_toolsuite is not installed. "
                "Install it with:  uv pip install oligo-designer-toolsuite  "
                "(or pip install oligo-designer-toolsuite), then restart the pipeline."
            )
        self.pipeline_type = pipeline_type.lower()
        self.output_dir = output_dir
        self.species = species
        self.annotation_source = annotation_source
        self.annotation_release = annotation_release
        self.genome_assembly = None  # Will be set during download
        self.taxon = taxon
        self.design_mode = design_mode.lower()
        self.write_intermediate_steps = write_intermediate_steps
        self.n_jobs = n_jobs
        
        # Validate design_mode
        if self.design_mode not in ['filter_only', 'complete']:
            raise ValueError(f"design_mode must be 'filter_only' or 'complete', got: {design_mode}")
        
        # Setup directories
        self.dirs = setup_directories(output_dir)
        
        # Stage 1 components
        self.ftp_loader = None
        self.region_generator = None
        
        # Stage 2 components
        self.pipeline = None
        
        # Paths to generated files
        self.gtf_file = None
        self.genome_file = None
        self.target_fasta = None
        self.reference_fasta = None
        
        print(f"Initialized Complete {pipeline_type.upper()} Probe Design Pipeline")
        print(f"  Species: {species}")
        print(f"  Source: {annotation_source} (release {annotation_release})")
        print(f"  Output: {self.dirs['base']}")
    
    
    # ========================================================================
    # STAGE 1: Reference Preparation
    # ========================================================================
    
    def download_references(self):
        """
        Download genome and annotation files.
        
        Returns:
        --------
        tuple : (gtf_file, genome_file)
        """
        print("\n" + "="*80)
        print("STAGE 1: PREPARING GENOMIC REFERENCES")
        print("="*80)
        print("\n[Step 1] Downloading reference files...")
        
        # Initialize FTP loader based on annotation source
        if self.annotation_source == 'ensembl':
            self.ftp_loader = FtpLoaderEnsembl(
                dir_output=str(self.dirs['references']),
                species=self.species,
                annotation_release=self.annotation_release
            )
        elif self.annotation_source == 'ncbi':
            self.ftp_loader = FtpLoaderNCBI(
                dir_output=str(self.dirs['references']),
                taxon=self.taxon,
                species=self.species,
                annotation_release=self.annotation_release
            )
        else:
            raise ValueError(f"Unknown annotation source: {self.annotation_source}")
        
        # Download files using the correct API
        print("  - Downloading GTF annotation...")
        self.gtf_file, downloaded_release, downloaded_assembly = self.ftp_loader.download_files(file_type='gtf')
        
        # Update metadata only if download provides valid values
        if downloaded_release is not None:
            self.annotation_release = downloaded_release
        if downloaded_assembly is not None:
            self.genome_assembly = downloaded_assembly
        
        print("  - Downloading genome FASTA...")
        self.genome_file, _, _ = self.ftp_loader.download_files(file_type='fasta')
        
        print(f"  Metadata: release={self.annotation_release}, assembly={self.genome_assembly}")
        
        print(f"\n✓ GTF: {self.gtf_file}")
        print(f"✓ Genome: {self.genome_file}")
        
        return self.gtf_file, self.genome_file
    
    
    def extract_gene_sequences(
        self,
        gene_list: List[str],
        region_type: str = "transcript",
        include_utr: bool = True,
        auto_convert_symbols: bool = True
    ):
        """
        Extract sequences for target genes.
        
        Parameters:
        -----------
        gene_list : list
            List of gene symbols or gene IDs (e.g., ['Gapdh', 'Actb'] or ['ENSMUSG00000057666'])
            If auto_convert_symbols=True, will automatically convert symbols to IDs
        region_type : str
            'transcript', 'exon', 'cds', etc.
        include_utr : bool
            Include UTR regions
        auto_convert_symbols : bool
            If True, automatically detect and convert gene symbols to IDs using GTF
        
        Returns:
        --------
        list : converted gene IDs (or original list if no conversion needed)
        """
        print(f"\n[Step 2] Extracting sequences for {len(gene_list)} genes...")
        print(f"  Region type: {region_type}")
        
        # Auto-convert gene symbols to IDs if needed
        if auto_convert_symbols:
            # Check if first gene looks like a symbol (not an ID)
            first_gene = gene_list[0]
            if not (first_gene.startswith('ENS') or first_gene.startswith('XM_') or first_gene.startswith('NM_')):
                print(f"  Detected gene symbols (e.g., '{first_gene}'), converting to IDs...")
                symbol_to_id = convert_gene_symbols_to_ids(gene_list, self.gtf_file, self.annotation_source)
                
                # Convert list to IDs
                gene_ids = [symbol_to_id.get(symbol, symbol) for symbol in gene_list]
                
                # Store the mapping for later reverse conversion
                self._symbol_to_id_mapping = symbol_to_id
                
                # Warn about any unconverted symbols
                unconverted = [g for g, gid in zip(gene_list, gene_ids) if g == gid and g not in symbol_to_id]
                if unconverted:
                    print(f"\n  WARNING: {len(unconverted)} genes could not be converted and will be used as-is:")
                    for gene in unconverted[:5]:
                        print(f"    {gene}")
                
                gene_list = gene_ids
            else:
                print(f"  Detected gene IDs (e.g., '{first_gene}'), using as-is...")
        
        # Store the converted gene IDs for later use
        self.current_gene_ids = gene_list
        
        # Store the converted gene IDs for later use
        self.current_gene_ids = gene_list
        
        # Initialize region generator
        self.region_generator = CustomGenomicRegionGenerator(
            annotation_file=self.gtf_file,
            sequence_file=self.genome_file,
            files_source=self.annotation_source,
            species=self.species,
            annotation_release=self.annotation_release,
            genome_assembly=self.genome_assembly,
            dir_output=str(self.dirs['sequences'])
        )
        
        # Generate sequences based on region type
        # These methods generate sequences for ALL genes in the annotation
        print("  Generating sequences (this requires bedtools to be installed)...")
        if region_type == "transcript" or region_type == "gene":
            all_sequences_file = self.region_generator.get_sequence_gene()
        elif region_type == "exon":
            all_sequences_file = self.region_generator.get_sequence_exon()
        elif region_type == "cds":
            all_sequences_file = self.region_generator.get_sequence_CDS()
        else:
            raise ValueError(f"Unsupported region type: {region_type}")
        
        # Check if the file was actually created
        from pathlib import Path
        if not Path(all_sequences_file).exists():
            raise FileNotFoundError(
                f"Sequence file was not generated: {all_sequences_file}\n\n"
                "This usually means bedtools is not installed or failed.\n"
                "Please install bedtools:\n"
                "  - On Ubuntu/Debian: sudo apt-get install bedtools\n"
                "  - On macOS: brew install bedtools\n"
                "  - On conda: conda install -c bioconda bedtools"
            )
        
        # Filter to keep only target genes
        from Bio import SeqIO
        print("  Filtering sequences for target genes...")
        target_fasta_path = self.dirs['sequences'] / 'target_genes.fasta'
        reference_fasta_path = self.dirs['sequences'] / 'reference_transcriptome.fasta'
        
        with open(target_fasta_path, 'w') as target_out:
            with open(reference_fasta_path, 'w') as ref_out:
                for record in SeqIO.parse(all_sequences_file, "fasta"):
                    # Extract gene_id from header (format: >gene_id::additional_info::coordinates)
                    gene_id = record.id.split("::")[0]
                    
                    # Write to reference (all genes)
                    SeqIO.write(record, ref_out, "fasta")
                    
                    # Write to target if in gene_list
                    if gene_id in gene_list:
                        SeqIO.write(record, target_out, "fasta")
        
        self.target_fasta = str(target_fasta_path)
        self.reference_fasta = str(reference_fasta_path)
        
        print(f"\n✓ Target sequences: {self.target_fasta}")
        print(f"✓ Reference transcriptome: {self.reference_fasta}")
        
        return gene_list  # Return converted gene IDs
    
    
    # ========================================================================
    # STAGE 2: Pipeline-Specific Probe Design
    # ========================================================================
    
    def initialize_pipeline(self):
        """Initialize the selected probe design pipeline."""
        print("\n" + "="*80)
        print(f"STAGE 2: RUNNING {self.pipeline_type.upper()} PROBE DESIGN")
        print("="*80)
        
        pipeline_dir = self.dirs['results'] / self.pipeline_type
        pipeline_dir.mkdir(exist_ok=True)
        
        # Workaround for ODT bug where set_developer_parameters is called in __init__
        # with already-converted parameters. We catch the error and manually set correct parameters.
        try:
            if self.pipeline_type == "scrinshot":
                self.pipeline = ScrinshotProbeDesigner(
                    write_intermediate_steps=self.write_intermediate_steps,
                    dir_output=str(pipeline_dir),
                    n_jobs=self.n_jobs
                )
            elif self.pipeline_type == "merfish":
                self.pipeline = MerfishProbeDesigner(
                    write_intermediate_steps=self.write_intermediate_steps,
                    dir_output=str(pipeline_dir),
                    n_jobs=self.n_jobs
                )
            elif self.pipeline_type == "seqfishplus":
                self.pipeline = SeqFishPlusProbeDesigner(
                    write_intermediate_steps=self.write_intermediate_steps,
                    dir_output=str(pipeline_dir),
                    n_jobs=self.n_jobs
                )
        except TypeError as e:
            if "attribute name must be string" in str(e):
                print("✓ Detected ODT parameter issue - applying workaround...")
                # The pipeline tried to convert parameters that are already dicts
                # This is a known issue in some ODT versions. Re-initialize manually.
                self._initialize_pipeline_with_workaround(pipeline_dir)
            else:
                raise
        
        print(f"✓ {self.pipeline_type.upper()} pipeline initialized")
    
    
    def _initialize_pipeline_with_workaround(self, pipeline_dir):
        """
        Workaround for ODT bug where Tm parameters are already dict objects.
        Temporarily patches set_developer_parameters to skip the problematic conversion.
        """
        from Bio.SeqUtils import MeltingTemp as mt
        
        if self.pipeline_type == "scrinshot":
            from oligo_designer_toolsuite.pipelines import ScrinshotProbeDesigner
            
            # Save the original method
            original_set_developer_parameters = ScrinshotProbeDesigner.set_developer_parameters
            
            # Create a patched version that handles already-converted dicts
            def patched_set_developer_parameters(self):
                """Patched version that doesn't re-convert dict objects"""
                # Try calling the original method - it might fail if dicts are already converted
                try:
                    original_set_developer_parameters(self)
                except (TypeError, AttributeError) as e:
                    # Original method failed, likely because values are already dicts
                    # We'll handle the conversion ourselves below
                    pass
                
                # Safely convert string references to dict objects
                # Only convert if: 1) attribute exists, 2) value is a string (not dict)
                if hasattr(self, 'target_probe_Tm_parameters'):
                    for param_name in ["nn_table", "tmm_table", "imm_table", "de_table"]:
                        if param_name in self.target_probe_Tm_parameters:
                            value = self.target_probe_Tm_parameters[param_name]
                            # Only convert if it's a string (not already a dict)
                            if isinstance(value, str):
                                self.target_probe_Tm_parameters[param_name] = getattr(mt, value)
                
                if hasattr(self, 'detection_oligo_Tm_parameters'):
                    for param_name in ["nn_table", "tmm_table", "imm_table", "de_table"]:
                        if param_name in self.detection_oligo_Tm_parameters:
                            value = self.detection_oligo_Tm_parameters[param_name]
                            # Only convert if it's a string (not already a dict)
                            if isinstance(value, str):
                                self.detection_oligo_Tm_parameters[param_name] = getattr(mt, value)
            
            try:
                # Temporarily replace the method
                ScrinshotProbeDesigner.set_developer_parameters = patched_set_developer_parameters
                
                # Now initialize normally - this should work
                self.pipeline = ScrinshotProbeDesigner(
                    dir_output=pipeline_dir,
                    write_intermediate_steps=self.write_intermediate_steps,
                    n_jobs=self.n_jobs
                )
            finally:
                # Restore the original method
                ScrinshotProbeDesigner.set_developer_parameters = original_set_developer_parameters
        else:
            raise NotImplementedError(f"Workaround not implemented for {self.pipeline_type}")
    
    
    def run_scrinshot_design(
        self,
        gene_ids: List[str],
        **kwargs
    ):
        """Run SCRINSHOT probe design (padlock probes)."""
        if self.pipeline is None:
            self.initialize_pipeline()
        
        if self.design_mode == 'filter_only':
            print("\n[Running SCRINSHOT design - FILTER ONLY mode...]")
            print("  (Only target probe design will run)")
        else:
            print("\n[Running SCRINSHOT design - COMPLETE mode...]")
        
        # Step 1: Design target probes (ALWAYS RUN - this is the filtering step)
        total_steps = 1 if self.design_mode == 'filter_only' else 4
        print(f"  [1/{total_steps}] Designing target probes...")
        target_probe_database = self.pipeline.design_target_probes(
            files_fasta_target_probe_database=[self.target_fasta],
            files_fasta_reference_database_targe_probe=[self.reference_fasta],  # Note: typo in ODT library
            gene_ids=gene_ids,
            target_probe_length_min=kwargs.get('target_probe_length_min', 40),
            target_probe_length_max=kwargs.get('target_probe_length_max', 45),
            set_size_opt=kwargs.get('set_size_opt', 5),
            set_size_min=kwargs.get('set_size_min', 3),
            n_sets=kwargs.get('n_sets', 100),
        )
        
        # For filter_only mode, skip steps 2-4 and return early
        if self.design_mode == 'filter_only':
            oligo_database = target_probe_database
        else:
            # Step 2: Design detection oligos (COMPLETE MODE ONLY)
            print("  [2/4] Designing detection oligos...")
            oligo_database = self.pipeline.design_detection_oligos(
                oligo_database=target_probe_database,
                detection_oligo_length_min=kwargs.get('detection_oligo_length_min', 15),
                detection_oligo_length_max=kwargs.get('detection_oligo_length_max', 40),
                detection_oligo_Tm_opt=kwargs.get('detection_oligo_Tm_opt', 56),
                **{k: v for k, v in kwargs.items() if k.startswith('detection_oligo_')}
            )
            
            # Step 3: Design padlock backbone (COMPLETE MODE ONLY)
            print("  [3/4] Designing padlock backbone...")
            oligo_database = self.pipeline.design_padlock_backbone(
                oligo_database=oligo_database
            )
            
            # Step 4: Generate output (COMPLETE MODE ONLY)
            print("  [4/4] Generating output files...")
            self.pipeline.generate_output(
                oligo_database=oligo_database,
                top_n_sets=kwargs.get('top_n_sets', 3)
            )
        
        # Extract successful genes (those with probe sets)
        successful_gene_ids = self._extract_successful_genes(oligo_database)
        successful_gene_symbols = self._convert_ids_to_symbols(successful_gene_ids)
        
        mode_msg = "candidate genes" if self.design_mode == 'filter_only' else "probes"
        print(f"\n✓ Successfully designed {mode_msg} for {len(successful_gene_ids)}/{len(gene_ids)} genes")
        if len(successful_gene_ids) < len(gene_ids):
            failed_ids = set(gene_ids) - set(successful_gene_ids)
            failed_symbols = self._convert_ids_to_symbols(list(failed_ids))
            print(f"  Failed genes: {', '.join(failed_symbols)}")
        
        return {
            'oligo_database': oligo_database,
            'output_dir': self.dirs['results'] / self.pipeline_type,
            'successful_gene_ids': successful_gene_ids,
            'successful_gene_symbols': successful_gene_symbols,
            'n_successful': len(successful_gene_ids),
            'n_requested': len(gene_ids),
            'design_mode': self.design_mode
        }
    
    
    def _extract_successful_genes(self, oligo_database):
        """Extract gene IDs that successfully generated probe sets."""
        successful_genes = set()
        
        # The OligoDatabase stores regions (genes) that have oligo sets
        # Access the database regions to find which genes have probes
        if hasattr(oligo_database, 'database'):
            # Check if database has regions attribute (might be LRUPickleDict cache)
            if hasattr(oligo_database.database, 'regions'):
                for region_id in oligo_database.database.regions:
                    successful_genes.add(region_id)
            # Fallback: try to access regions via dict keys if it's a dict-like object
            elif hasattr(oligo_database.database, 'keys'):
                for region_id in oligo_database.database.keys():
                    successful_genes.add(region_id)
            # Last resort: try direct iteration
            else:
                try:
                    for region_id in oligo_database.database:
                        successful_genes.add(region_id)
                except (TypeError, AttributeError) as e:
                    print(f"  Warning: Could not extract successful genes from database: {e}")
        
        return sorted(list(successful_genes))
    
    
    def _convert_ids_to_symbols(self, gene_ids: List[str]) -> List[str]:
        """Convert gene IDs back to symbols using the GTF annotation."""
        if not hasattr(self, 'gtf_file') or self.gtf_file is None:
            print("  Warning: No GTF file available for ID-to-symbol conversion")
            return gene_ids
        
        # Use the reverse mapping from the stored conversion
        if hasattr(self, '_symbol_to_id_mapping'):
            id_to_symbol = {v: k for k, v in self._symbol_to_id_mapping.items()}
            return [id_to_symbol.get(gene_id, gene_id) for gene_id in gene_ids]
        
        # Otherwise, parse GTF to create mapping
        print("  Converting gene IDs to symbols...")
        id_to_symbol = {}
        
        import gzip
        open_func = gzip.open if self.gtf_file.endswith('.gz') else open
        
        with open_func(self.gtf_file, 'rt') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                
                parts = line.strip().split('\t')
                if len(parts) < 9:
                    continue
                
                if parts[2] == 'gene':
                    attrs = parts[8]
                    gene_id = None
                    gene_name = None
                    
                    for attr in attrs.split(';'):
                        attr = attr.strip()
                        if attr.startswith('gene_id'):
                            gene_id = attr.split('"')[1]
                        elif attr.startswith('gene_name'):
                            gene_name = attr.split('"')[1]
                    
                    if gene_id and gene_name:
                        id_to_symbol[gene_id] = gene_name
        
        return [id_to_symbol.get(gene_id, gene_id) for gene_id in gene_ids]
    
    
    def run_merfish_design(
        self,
        gene_ids: List[str],
        **kwargs
    ):
        """Run MERFISH probe design (encoding probes with barcodes)."""
        if self.pipeline is None:
            self.initialize_pipeline()
        
        if self.design_mode == 'filter_only':
            print("\n[Running MERFISH design - FILTER ONLY mode...]")
            print("  (Only target probe design will run)")
        else:
            print("\n[Running MERFISH design - COMPLETE mode...]")
        
        # Step 1: Design target probes (ALWAYS RUN - this is the filtering step)
        total_steps = 1 if self.design_mode == 'filter_only' else 5
        print(f"  [1/{total_steps}] Designing target probes...")
        target_probe_database = self.pipeline.design_target_probes(
            files_fasta_target_probe_database=[self.target_fasta],
            files_fasta_reference_database_targe_probe=[self.reference_fasta],  # Note: typo in ODT library
            gene_ids=gene_ids,
            target_probe_length_min=kwargs.get('target_probe_length_min', 30),
            target_probe_length_max=kwargs.get('target_probe_length_max', 30),
            set_size_opt=kwargs.get('set_size_opt', 50),
            set_size_min=kwargs.get('set_size_min', 50),
            n_sets=kwargs.get('n_sets', 100),
            **{k: v for k, v in kwargs.items() if k.startswith('target_probe_')}
        )
        
        # For filter_only mode, skip steps 2-5 and return early
        if self.design_mode == 'filter_only':
            # Extract successful genes
            successful_gene_ids = self._extract_successful_genes(target_probe_database)
            successful_gene_symbols = self._convert_ids_to_symbols(successful_gene_ids)
            
            print(f"\n✓ Successfully identified {len(successful_gene_ids)}/{len(gene_ids)} candidate genes")
            if len(successful_gene_ids) < len(gene_ids):
                failed_ids = set(gene_ids) - set(successful_gene_ids)
                failed_symbols = self._convert_ids_to_symbols(list(failed_ids))
                print(f"  Failed genes: {', '.join(failed_symbols)}")
            
            return {
                'target_probe_database': target_probe_database,
                'output_dir': self.dirs['results'] / self.pipeline_type,
                'successful_gene_ids': successful_gene_ids,
                'successful_gene_symbols': successful_gene_symbols,
                'n_successful': len(successful_gene_ids),
                'n_requested': len(gene_ids),
                'design_mode': 'filter_only'
            }
        
        # COMPLETE MODE: Continue with steps 2-5
        # Step 2: Design readout probes
        print("  [2/5] Designing readout probes and codebook...")
        codebook, readout_probe_table = self.pipeline.design_readout_probes(
            n_genes=len(target_probe_database.database),
            files_fasta_reference_database_readout_probe=[self.reference_fasta],
            readout_probe_length=kwargs.get('readout_probe_length', 20),
            n_bits=kwargs.get('n_bits', 16),
            min_hamming_dist=kwargs.get('min_hamming_dist', 4),
            **{k: v for k, v in kwargs.items() if k.startswith('readout_probe_')}
        )
        
        # Step 3: Design encoding probes
        print("  [3/5] Combining target and readout probes...")
        encoding_probe_database = self.pipeline.design_encoding_probe(
            target_probe_database=target_probe_database,
            codebook=codebook,
            readout_probe_table=readout_probe_table
        )
        
        # Step 4: Design primers
        print("  [4/5] Designing primers...")
        reverse_primer, forward_primer = self.pipeline.design_primers(
            encoding_probe_database=encoding_probe_database,
            files_fasta_reference_database_primer=[self.reference_fasta],
            reverse_primer_sequence=kwargs.get('reverse_primer_sequence', 'CCCTATAGTGAGTCGTATTA'),
            primer_length=kwargs.get('primer_length', 20),
            **{k: v for k, v in kwargs.items() if k.startswith('primer_')}
        )
        
        # Step 5: Generate output
        print("  [5/5] Generating output files...")
        self.pipeline.generate_output(
            encoding_probe_database=encoding_probe_database,
            reverse_primer_sequence=reverse_primer,
            forward_primer_sequence=forward_primer,
            top_n_sets=kwargs.get('top_n_sets', 3)
        )
        
        # Extract successful genes
        successful_gene_ids = self._extract_successful_genes(encoding_probe_database)
        successful_gene_symbols = self._convert_ids_to_symbols(successful_gene_ids)
        
        print(f"\n✓ Successfully designed complete probes for {len(successful_gene_ids)}/{len(gene_ids)} genes")
        
        return {
            'encoding_probe_database': encoding_probe_database,
            'codebook': codebook,
            'readout_probe_table': readout_probe_table,
            'reverse_primer': reverse_primer,
            'forward_primer': forward_primer,
            'output_dir': self.dirs['results'] / self.pipeline_type,
            'successful_gene_ids': successful_gene_ids,
            'successful_gene_symbols': successful_gene_symbols,
            'n_successful': len(successful_gene_ids),
            'n_requested': len(gene_ids),
            'design_mode': 'complete'
        }
    
    
    def run_seqfishplus_design(
        self,
        gene_ids: List[str],
        **kwargs
    ):
        """Run seqFISH+ probe design (encoding probes with pseudo-colors)."""
        if self.pipeline is None:
            self.initialize_pipeline()
        
        if self.design_mode == 'filter_only':
            print("\n[Running seqFISH+ design - FILTER ONLY mode...]")
            print("  (Only target probe design will run)")
        else:
            print("\n[Running seqFISH+ design - COMPLETE mode...]")
        
        # Step 1: Design target probes (ALWAYS RUN - this is the filtering step)
        total_steps = 1 if self.design_mode == 'filter_only' else 5
        print(f"  [1/{total_steps}] Designing target probes...")
        target_probe_database = self.pipeline.design_target_probes(
            files_fasta_target_probe_database=[self.target_fasta],
            files_fasta_reference_database_targe_probe=[self.reference_fasta],  # Note: typo in ODT library
            gene_ids=gene_ids,
            target_probe_length_min=kwargs.get('target_probe_length_min', 28),
            target_probe_length_max=kwargs.get('target_probe_length_max', 28),
            set_size_opt=kwargs.get('set_size_opt', 24),
            set_size_min=kwargs.get('set_size_min', 24),
            distance_between_target_probes=kwargs.get('distance_between_target_probes', 2),
            n_sets=kwargs.get('n_sets', 100),
            **{k: v for k, v in kwargs.items() if k.startswith('target_probe_')}
        )
        
        # For filter_only mode, skip steps 2-5 and return early
        if self.design_mode == 'filter_only':
            # Extract successful genes
            successful_gene_ids = self._extract_successful_genes(target_probe_database)
            successful_gene_symbols = self._convert_ids_to_symbols(successful_gene_ids)
            
            print(f"\n✓ Successfully identified {len(successful_gene_ids)}/{len(gene_ids)} candidate genes")
            if len(successful_gene_ids) < len(gene_ids):
                failed_ids = set(gene_ids) - set(successful_gene_ids)
                failed_symbols = self._convert_ids_to_symbols(list(failed_ids))
                print(f"  Failed genes: {', '.join(failed_symbols)}")
            
            return {
                'target_probe_database': target_probe_database,
                'output_dir': self.dirs['results'] / self.pipeline_type,
                'successful_gene_ids': successful_gene_ids,
                'successful_gene_symbols': successful_gene_symbols,
                'n_successful': len(successful_gene_ids),
                'n_requested': len(gene_ids),
                'design_mode': 'filter_only'
            }
        
        # COMPLETE MODE: Continue with steps 2-5
        # Step 2: Design readout probes
        print("  [2/5] Designing readout probes and codebook...")
        codebook, readout_probe_table = self.pipeline.design_readout_probes(
            n_genes=len(target_probe_database.database),
            files_fasta_reference_database_readout_probe=[self.reference_fasta],
            readout_probe_length=kwargs.get('readout_probe_length', 15),
            n_barcode_rounds=kwargs.get('n_barcode_rounds', 4),
            n_pseudocolors=kwargs.get('n_pseudocolors', 20),
            **{k: v for k, v in kwargs.items() if k.startswith('readout_probe_')}
        )
        
        # Step 3: Design encoding probes
        print("  [3/5] Combining target and readout probes...")
        encoding_probe_database = self.pipeline.design_encoding_probe(
            target_probe_database=target_probe_database,
            codebook=codebook,
            readout_probe_table=readout_probe_table
        )
        
        # Step 4: Design primers
        print("  [4/5] Designing primers...")
        reverse_primer, forward_primer = self.pipeline.design_primers(
            encoding_probe_database=encoding_probe_database,
            files_fasta_reference_database_primer=[self.reference_fasta],
            reverse_primer_sequence=kwargs.get('reverse_primer_sequence', 'CCCTATAGTGAGTCGTATTA'),
            primer_length=kwargs.get('primer_length', 20),
            **{k: v for k, v in kwargs.items() if k.startswith('primer_')}
        )
        
        # Step 5: Generate output
        print("  [5/5] Generating output files...")
        self.pipeline.generate_output(
            encoding_probe_database=encoding_probe_database,
            reverse_primer_sequence=reverse_primer,
            forward_primer_sequence=forward_primer,
            top_n_sets=kwargs.get('top_n_sets', 3)
        )
        
        # Extract successful genes
        successful_gene_ids = self._extract_successful_genes(encoding_probe_database)
        successful_gene_symbols = self._convert_ids_to_symbols(successful_gene_ids)
        
        print(f"\n✓ Successfully designed complete probes for {len(successful_gene_ids)}/{len(gene_ids)} genes")
        
        return {
            'encoding_probe_database': encoding_probe_database,
            'codebook': codebook,
            'readout_probe_table': readout_probe_table,
            'reverse_primer': reverse_primer,
            'forward_primer': forward_primer,
            'output_dir': self.dirs['results'] / self.pipeline_type,
            'successful_gene_ids': successful_gene_ids,
            'successful_gene_symbols': successful_gene_symbols,
            'n_successful': len(successful_gene_ids),
            'n_requested': len(gene_ids),
            'design_mode': 'complete'
        }
    
    
    # ========================================================================
    # CONVENIENCE METHOD: Run Complete Pipeline
    # ========================================================================
    
    def run_complete_pipeline(
        self,
        gene_list: List[str],
        region_type: str = "transcript",
        **kwargs
    ):
        """
        Run the complete two-stage pipeline.
        
        The behavior depends on the design_mode parameter set during initialization:
        - 'filter_only': Run only target probe design to identify designable genes (fast)
        - 'complete': Run full pipeline including detection/readout/primers (slow)
        
        Parameters:
        -----------
        gene_list : list
            List of gene symbols/IDs to design probes for
        region_type : str
            Region type to extract ('transcript', 'exon', 'cds')
        **kwargs : dict
            Additional parameters for probe design (method-specific)
        
        Returns:
        --------
        dict : Results dictionary with probe database and output paths
               For 'filter_only': returns successful_gene_ids/symbols
               For 'complete': returns full probe databases, primers, codebooks
        """
        print("\n" + "="*80)
        print("COMPLETE PROBE DESIGN PIPELINE")
        print(f"Method: {self.pipeline_type.upper()}")
        print(f"Genes: {len(gene_list)}")
        print("="*80)
        
        # Stage 1: Prepare references and extract sequences
        if self.gtf_file is None or self.genome_file is None:
            self.download_references()
        
        if self.target_fasta is None or self.reference_fasta is None:
            gene_ids = self.extract_gene_sequences(gene_list, region_type)
        else:
            # If sequences already exist, use stored gene IDs or original list
            gene_ids = getattr(self, 'current_gene_ids', gene_list)
        
        # Stage 2: Run pipeline-specific design (use converted gene IDs)
        if self.pipeline_type == "scrinshot":
            results = self.run_scrinshot_design(gene_ids, **kwargs)
        elif self.pipeline_type == "merfish":
            results = self.run_merfish_design(gene_ids, **kwargs)
        elif self.pipeline_type == "seqfishplus":
            results = self.run_seqfishplus_design(gene_ids, **kwargs)
        else:
            raise ValueError(f"Unknown pipeline: {self.pipeline_type}")
        
        print("\n" + "="*80)
        print("✓ PIPELINE COMPLETE!")
        print("="*80)
        print(f"\nOutput directory: {results['output_dir']}")
        print("\nNext steps:")
        print("1. Review the generated probe sequences")
        print("2. Check the codebook/barcode assignments")
        print("3. Order probes for synthesis")
        
        return results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    
    # Example genes
    candidate_genes = [
        'Gapdh', 'Actb', 'Pdgfra', 'Sox2', 'Pecam1',
        'Ptprc', 'Epcam', 'Vim', 'Krt8', 'Krt18'
    ]
    
    # ========================================================================
    # Example 1: Gene Filtering (Filter Only Mode)
    # ========================================================================
    print("\n" + "="*80)
    print("EXAMPLE 1: GENE FILTERING - Identify Designable Genes")
    print("="*80)
    
    filter_pipeline = CompleteProbeDesignPipeline(
        pipeline_type="scrinshot",
        output_dir="./output_filter_only",
        species="mus_musculus",
        annotation_source="ensembl",
        annotation_release="110",
        design_mode="filter_only",  # Only run target probe design
        n_jobs=4
    )
    
    filter_results = filter_pipeline.run_complete_pipeline(
        gene_list=candidate_genes,
        set_size_opt=5,
        set_size_min=3
    )
    
    print(f"\n✓ Designable genes: {filter_results['successful_gene_symbols']}")
    print(f"✓ Success rate: {filter_results['n_successful']}/{filter_results['n_requested']}")
    
    # Use only designable genes in your downstream selection pipeline
    usable_genes = filter_results['successful_gene_symbols']
    
    # ========================================================================
    # Example 2: Complete MERFISH Pipeline (Full Probe Design)
    # ========================================================================
    print("\n" + "="*80)
    print("EXAMPLE 2: COMPLETE MERFISH PIPELINE - Ready-to-Use Probes")
    print("="*80)
    
    merfish_pipeline = CompleteProbeDesignPipeline(
        pipeline_type="merfish",
        output_dir="./output_merfish_complete",
        species="mus_musculus",
        annotation_source="ensembl",
        annotation_release="110",
        design_mode="complete",  # Run all steps for final probes
        n_jobs=4
    )
    
    merfish_results = merfish_pipeline.run_complete_pipeline(
        gene_list=candidate_genes,
        # Pipeline-specific parameters
        set_size_opt=50,
        n_bits=16,
        top_n_sets=3
    )
    
    # ========================================================================
    # Example 3: Step-by-step control
    # ========================================================================
    print("\n" + "="*80)
    print("EXAMPLE 3: STEP-BY-STEP seqFISH+ PIPELINE")
    print("="*80)
    
    seqfish_pipeline = CompleteProbeDesignPipeline(
        pipeline_type="seqfishplus",
        output_dir="./output_seqfish_complete",
        species="mus_musculus",
        design_mode="complete",
        n_jobs=4
    )
    
    # Stage 1: Prepare sequences
    seqfish_pipeline.download_references()
    seqfish_pipeline.extract_gene_sequences(candidate_genes)
    
    # Stage 2: Design probes
    seqfish_results = seqfish_pipeline.run_seqfishplus_design(
        gene_ids=candidate_genes,
        set_size_opt=24,
        n_barcode_rounds=4,
        top_n_sets=3
    )

