"""
This file contains some functions copied from the original tangram source code. This had to be done to change the behaviour around non expressed genes.
For the purpose of benchmarking the entire count matrix of shared genes between a query and reference dataset is reconstructed
However the functions pp_adatas, map_cells_to_space and project genes from the original tangram source code either filter 
zero count genes or raise an Error when it encounters them. This behaviour was adjusted in the copied functions.
The concrete changes can be found via ctrl/cmd + f "code adapted"
"""

from dataclasses import dataclass
import logging
from typing import Optional

import numpy as np
import pandas as pd
from tangram import mapping_utils as mu
from tangram import utils as ut
from tangram import mapping_optimizer as mo
from numpy.typing import NDArray
from numpy import number
from anndata import AnnData
import scanpy as sc
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
import torch

def pp_adatas_unfiltered(adata_sc, adata_sp, genes=None, gene_to_lowercase = True):
    """
    Pre-process AnnDatas so that they can be mapped. Specifically:
    - Remove genes that all entries are zero
    - Find the intersection between adata_sc, adata_sp and given marker gene list, save the intersected markers in two adatas
    - Calculate density priors and save it with adata_sp

    Args:
        adata_sc (AnnData): single cell data
        adata_sp (AnnData): spatial expression data
        genes (List): Optional. List of genes to use. If `None`, all genes are used.
    
    Returns:
        update adata_sc by creating `uns` `training_genes` `overlap_genes` fields 
        update adata_sp by creating `uns` `training_genes` `overlap_genes` fields and creating `obs` `rna_count_based_density` & `uniform_density` field
    """

    # remove all-zero-valued genes <- part of the original source code, commented out
    #sc.pp.filter_genes(adata_sc, min_cells=1) code adapted
    #sc.pp.filter_genes(adata_sp, min_cells=1) code adapted

    if genes is None:
        # Use all genes
        genes = adata_sc.var.index
               
    # put all var index to lower case to align
    if gene_to_lowercase:
        adata_sc.var.index = [g.lower() for g in adata_sc.var.index]
        adata_sp.var.index = [g.lower() for g in adata_sp.var.index]
        genes = list(g.lower() for g in genes)

    adata_sc.var_names_make_unique()
    adata_sp.var_names_make_unique()
    

    # Refine `marker_genes` so that they are shared by both adatas
    genes = list(set(genes) & set(adata_sc.var.index) & set(adata_sp.var.index))
    # logging.info(f"{len(genes)} shared marker genes.")

    adata_sc.uns["training_genes"] = genes
    adata_sp.uns["training_genes"] = genes
    logging.info(
        "{} training genes are saved in `uns``training_genes` of both single cell and spatial Anndatas.".format(
            len(genes)
        )
    )

    # Find overlap genes between two AnnDatas
    overlap_genes = list(set(adata_sc.var.index) & set(adata_sp.var.index))
    # logging.info(f"{len(overlap_genes)} shared genes.")

    adata_sc.uns["overlap_genes"] = overlap_genes
    adata_sp.uns["overlap_genes"] = overlap_genes
    logging.info(
        "{} overlapped genes are saved in `uns``overlap_genes` of both single cell and spatial Anndatas.".format(
            len(overlap_genes)
        )
    )

    # Calculate uniform density prior as 1/number_of_spots
    adata_sp.obs["uniform_density"] = np.ones(adata_sp.X.shape[0]) / adata_sp.X.shape[0]
    logging.info(
        f"uniform based density prior is calculated and saved in `obs``uniform_density` of the spatial Anndata."
    )

    # Calculate rna_count_based density prior as % of rna molecule count
    rna_count_per_spot = np.array(adata_sp.X.sum(axis=1)).squeeze()
    adata_sp.obs["rna_count_based_density"] = rna_count_per_spot / np.sum(rna_count_per_spot)
    logging.info(
        f"rna count based density prior is calculated and saved in `obs``rna_count_based_density` of the spatial Anndata."
    )

def map_cells_to_space(
    adata_sc,
    adata_sp,
    cv_train_genes=None,
    cluster_label=None,
    mode="cells",
    device="cpu",
    learning_rate=0.1,
    num_epochs=1000,
    scale=True,
    lambda_d=0,
    lambda_g1=1,
    lambda_g2=0,
    lambda_r=0,
    lambda_count=1,
    lambda_f_reg=1,
    target_count=None,
    random_state=None,
    verbose=True,
    density_prior='rna_count_based',
):
    """
    Map single cell data (`adata_sc`) on spatial data (`adata_sp`).
    
    Args:
        adata_sc (AnnData): single cell data
        adata_sp (AnnData): gene spatial data
        cv_train_genes (list): Optional. Training gene list. Default is None.
        cluster_label (str): Optional. Field in `adata_sc.obs` used for aggregating single cell data. Only valid for `mode=clusters`.
        mode (str): Optional. Tangram mapping mode. Currently supported: 'cell', 'clusters', 'constrained'. Default is 'cell'.
        device (string or torch.device): Optional. Default is 'cpu'.
        learning_rate (float): Optional. Learning rate for the optimizer. Default is 0.1.
        num_epochs (int): Optional. Number of epochs. Default is 1000.
        scale (bool): Optional. Whether weight input single cell data by the number of cells in each cluster, only valid when cluster_label is not None. Default is True.
        lambda_d (float): Optional. Hyperparameter for the density term of the optimizer. Default is 0.
        lambda_g1 (float): Optional. Hyperparameter for the gene-voxel similarity term of the optimizer. Default is 1.
        lambda_g2 (float): Optional. Hyperparameter for the voxel-gene similarity term of the optimizer. Default is 0.
        lambda_r (float): Optional. Strength of entropy regularizer. An higher entropy promotes probabilities of each cell peaked over a narrow portion of space. lambda_r = 0 corresponds to no entropy regularizer. Default is 0.
        lambda_count (float): Optional. Regularizer for the count term. Default is 1. Only valid when mode == 'constrained'
        lambda_f_reg (float): Optional. Regularizer for the filter, which promotes Boolean values (0s and 1s) in the filter. Only valid when mode == 'constrained'. Default is 1.
        target_count (int): Optional. The number of cells to be filtered. Default is None.
        random_state (int): Optional. pass an int to reproduce training. Default is None.
        verbose (bool): Optional. If print training details. Default is True.
        density_prior (str, ndarray or None): Spatial density of spots, when is a string, value can be 'rna_count_based' or 'uniform', when is a ndarray, shape = (number_spots,). This array should satisfy the constraints sum() == 1. If None, the density term is ignored. Default value is 'rna_count_based'.

    Returns:
        a cell-by-spot AnnData containing the probability of mapping cell i on spot j.
        The `uns` field of the returned AnnData contains the training genes.
    """

    # check invalid values for arguments
    if lambda_g1 == 0:
        raise ValueError("lambda_g1 cannot be 0.")

    if (type(density_prior) is str) and (
        density_prior not in ["rna_count_based", "uniform", None]
    ):
        raise ValueError("Invalid input for density_prior.")

    if density_prior is not None and (lambda_d == 0 or lambda_d is None):
        lambda_d = 1

    if lambda_d > 0 and density_prior is None:
        raise ValueError("When lambda_d is set, please define the density_prior.")

    if mode not in ["clusters", "cells", "constrained"]:
        raise ValueError('Argument "mode" must be "cells", "clusters" or "constrained')

    if mode == "clusters" and cluster_label is None:
        raise ValueError("A cluster_label must be specified if mode is 'clusters'.")

    if mode == "constrained" and not all([target_count, lambda_f_reg, lambda_count]):
        raise ValueError(
            "target_count, lambda_f_reg and lambda_count must be specified if mode is 'constrained'."
        )

    if mode == "clusters":
        adata_sc = mu.adata_to_cluster_expression(
            adata_sc, cluster_label, scale, add_density=True
        )

    # Check if training_genes key exist/is valid in adatas.uns
    if not set(["training_genes", "overlap_genes"]).issubset(set(adata_sc.uns.keys())):
        raise ValueError("Missing tangram parameters. Run `pp_adatas()`.")

    if not set(["training_genes", "overlap_genes"]).issubset(set(adata_sp.uns.keys())):
        raise ValueError("Missing tangram parameters. Run `pp_adatas()`.")

    assert list(adata_sp.uns["training_genes"]) == list(adata_sc.uns["training_genes"])

    # get training_genes
    if cv_train_genes is None:
        training_genes = adata_sc.uns["training_genes"]
    elif cv_train_genes is not None:
        if set(cv_train_genes).issubset(set(adata_sc.uns["training_genes"])):
            training_genes = cv_train_genes
        else:
            raise ValueError(
                "Given training genes list should be subset of two AnnDatas."
            )

    logging.info("Allocate tensors for mapping.")
    # Allocate tensors (AnnData matrix can be sparse or not)

    if isinstance(adata_sc.X, csc_matrix) or isinstance(adata_sc.X, csr_matrix): # type: ignore
        S = np.array(adata_sc[:, training_genes].X.toarray(), dtype="float32",) # type: ignore
    elif isinstance(adata_sc.X, np.ndarray):
        S = np.array(adata_sc[:, training_genes].X.toarray(), dtype="float32",) # type: ignore
    else:
        X_type = type(adata_sc.X)
        logging.error("AnnData X has unrecognized type: {}".format(X_type))
        raise NotImplementedError

    if isinstance(adata_sp.X, csc_matrix) or isinstance(adata_sp.X, csr_matrix): # type: ignore
        G = np.array(adata_sp[:, training_genes].X.toarray(), dtype="float32")
    elif isinstance(adata_sp.X, np.ndarray):
        G = np.array(adata_sp[:, training_genes].X, dtype="float32")
    else:
        X_type = type(adata_sp.X)
        logging.error("AnnData X has unrecognized type: {}".format(X_type))
        raise NotImplementedError

    #if not S.any(axis=0).all() or not G.any(axis=0).all(): code adapted
    #    raise ValueError("Genes with all zero values detected. Run `pp_adatas()`.") code adapted

    d_source = None

    # define density_prior if 'rna_count_based' is passed to the density_prior argument:
    d_str = density_prior
    if type(density_prior) is np.ndarray:
        d_str = "customized"

    if density_prior == "rna_count_based":
        density_prior = adata_sp.obs["rna_count_based_density"]

    # define density_prior if 'uniform' is passed to the density_prior argument:
    elif density_prior == "uniform":
        density_prior = adata_sp.obs["uniform_density"]

    if mode == "cells":
        d = density_prior

    if mode == "clusters":
        d_source = np.array(adata_sc.obs["cluster_density"])

    if mode in ["clusters", "constrained"]:
        if density_prior is None:
            d = adata_sp.obs["uniform_density"]
            d_str = "uniform"
        else:
            d = density_prior
        if lambda_d is None or lambda_d == 0:
            lambda_d = 1

    # Choose device
    device = torch.device(device)  # for gpu

    if verbose:
        print_each = 100
    else:
        print_each = None

    if mode in ["cells", "clusters"]:
        hyperparameters = {
            "lambda_d": lambda_d,  # KL (ie density) term
            "lambda_g1": lambda_g1,  # gene-voxel cos sim
            "lambda_g2": lambda_g2,  # voxel-gene cos sim
            "lambda_r": lambda_r,  # regularizer: penalize entropy
            "d_source": d_source,
        }

        logging.info(
            "Begin training with {} genes and {} density_prior in {} mode...".format(
                len(training_genes), d_str, mode
            )
        )
        mapper = mo.Mapper(
            S=S, G=G, d=d, device=device, random_state=random_state, **hyperparameters, # type: ignore
        )

        # TODO `train` should return the loss function

        mapping_matrix, training_history = mapper.train(
            learning_rate=learning_rate, num_epochs=num_epochs, print_each=print_each, # type: ignore
        )

    # constrained mode
    elif mode == "constrained":
        hyperparameters = {
            "lambda_d": lambda_d,  # KL (ie density) term
            "lambda_g1": lambda_g1,  # gene-voxel cos sim
            "lambda_g2": lambda_g2,  # voxel-gene cos sim
            "lambda_r": lambda_r,  # regularizer: penalize entropy
            "lambda_count": lambda_count,
            "lambda_f_reg": lambda_f_reg,
            "target_count": target_count,
        }

        logging.info(
            "Begin training with {} genes and {} density_prior in {} mode...".format(
                len(training_genes), d_str, mode
            )
        )

        mapper = mo.MapperConstrained(
            S=S, G=G, d=d, device=device, random_state=random_state, **hyperparameters, # type: ignore
        )

        mapping_matrix, F_out, training_history = mapper.train(
            learning_rate=learning_rate, num_epochs=num_epochs, print_each=print_each, # type: ignore
        )

    logging.info("Saving results..")
    adata_map = sc.AnnData(
        X=mapping_matrix,
        obs=adata_sc[:, training_genes].obs.copy(),
        var=adata_sp[:, training_genes].obs.copy(),
    )

    if mode == "constrained":
        adata_map.obs["F_out"] = F_out

    # Annotate cosine similarity of each training gene
    G_predicted = adata_map.X.T @ S # type: ignore
    cos_sims = []
    for v1, v2 in zip(G.T, G_predicted.T):
        norm_sq = np.linalg.norm(v1) * np.linalg.norm(v2)
        cos_sims.append((v1 @ v2) / norm_sq)

    df_cs = pd.DataFrame(cos_sims, training_genes, columns=["train_score"])
    df_cs = df_cs.sort_values(by="train_score", ascending=False)
    adata_map.uns["train_genes_df"] = df_cs

    # Annotate sparsity of each training genes
    ut.annotate_gene_sparsity(adata_sc)
    ut.annotate_gene_sparsity(adata_sp)
    adata_map.uns["train_genes_df"]["sparsity_sc"] = adata_sc[
        :, training_genes
    ].var.sparsity
    adata_map.uns["train_genes_df"]["sparsity_sp"] = adata_sp[
        :, training_genes
    ].var.sparsity
    adata_map.uns["train_genes_df"]["sparsity_diff"] = (
        adata_sp[:, training_genes].var.sparsity
        - adata_sc[:, training_genes].var.sparsity
    )

    adata_map.uns["training_history"] = training_history

    return adata_map

def project_genes_unfiltered(adata_map, adata_sc, cluster_label=None, scale=True):
    """
    Transfer gene expression from the single cell onto space.

    Args:
        adata_map (AnnData): single cell data
        adata_sp (AnnData): gene spatial data
        cluster_label (AnnData): Optional. Should be consistent with the 'cluster_label' argument passed to `map_cells_to_space` function.
        scale (bool): Optional. Should be consistent with the 'scale' argument passed to `map_cells_to_space` function.

    Returns:
        AnnData: spot-by-gene AnnData containing spatial gene expression from the single cell data.
    """

    # put all var index to lower case to align
    adata_sc.var.index = [g.lower() for g in adata_sc.var.index]

    # make varnames unique for adata_sc
    adata_sc.var_names_make_unique()

    # remove all-zero-valued genes
    #sc.pp.filter_genes(adata_sc, min_cells=1) code adapted

    if cluster_label:
        adata_sc = mu.adata_to_cluster_expression(adata_sc, cluster_label, scale=scale)

    if not adata_map.obs.index.equals(adata_sc.obs.index):
        raise ValueError("The two AnnDatas need to have same `obs` index.")
    if hasattr(adata_sc.X, "toarray"):
        adata_sc.X = adata_sc.X.toarray() # type: ignore
    X_space = adata_map.X.T @ adata_sc.X
    adata_ge = sc.AnnData(
        X=X_space, obs=adata_map.var, var=adata_sc.var, uns=adata_sc.uns
    )
    training_genes = adata_map.uns["train_genes_df"].index.values
    adata_ge.var["is_training"] = adata_ge.var.index.isin(training_genes)
    return adata_ge


@dataclass
class TangramPredictor:
    verbose: bool = False

    adata_reference: Optional[AnnData] = None
    n_shared_features: Optional[int] = None

    def fit(self, X: NDArray[number], y: NDArray[number]) -> "TangramPredictor":
        """
        Fits the Tangram predictor on the reference matrix.

        Parameters
        ----------
        X : ndarray
            Shared features (or full reference if y is None).
        y : ndarray, optional
            Predicted features. If provided, concatenates X and y.
        """
        reference_matrix = np.concatenate([X, y], axis=1)
        self.n_shared_features = X.shape[1]
        self.adata_reference = AnnData(reference_matrix)
        return self

    def predict(self, X: NDArray[number]) -> NDArray[number]:
        """
        Predicts the missing / non-shared features for the query matrix X.
        """
        if self.adata_reference is None or self.n_shared_features is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        adata_query = AnnData(X)
        adata_reference = self.adata_reference.copy()

        pp_adatas_unfiltered(adata_reference, adata_query)

        ad_map = map_cells_to_space(
            adata_sc=adata_reference,
            adata_sp=adata_query,
            verbose=self.verbose
        )

        ad_ge = project_genes_unfiltered(ad_map, adata_reference)
        assert isinstance(ad_ge.X, np.ndarray)

        predicted = ad_ge.X[:, self.n_shared_features:]

        return predicted