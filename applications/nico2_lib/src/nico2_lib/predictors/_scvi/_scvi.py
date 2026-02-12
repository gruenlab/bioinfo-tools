from anndata.typing import AnnData
from nico2_lib.predictors._scvi._model import recVAE
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import CategoricalJointObsField, CategoricalObsField, LayerField, NumericalJointObsField, NumericalObsField
from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin


class SCVI(UnsupervisedTrainingMixin, BaseModelClass):
    """single-cell Variational Inference [Lopez18]_."""

    def __init__(
        self,
        adata: AnnData,
        idx: list,
        n_latent: int = 10,
        beta: float = 1,
        C: float = 0,
        **model_kwargs,
    ):
        super().__init__(adata)

        # self.module = VAE(
        self.module = recVAE(
            n_input=self.summary_stats["n_vars"],
            # n_batch=self.summary_stats["n_batch"],
            idx=idx,
            n_latent=n_latent,
            beta=beta,
            C=C,
            **model_kwargs,
        )
        self._model_summary_string = (
            f"SCVI Model with the following params: \nn_latent: {n_latent}"
        )
        self.init_params_ = self._get_init_params(locals())

    @classmethod
    def setup_anndata(
        cls,
        adata: AnnData,
        batch_key: str | None = None,
        layer: str | None = None,
        **kwargs,
    ) -> AnnData | None:
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, None),
            NumericalObsField(REGISTRY_KEYS.SIZE_FACTOR_KEY, None, required=False),
            CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, None),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, None),
        ]
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)