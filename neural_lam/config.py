"""Configuration dataclasses and helpers for Neural-LAM experiments."""

# Standard library
import dataclasses
from pathlib import Path
from typing import cast

# Third-party
import dataclass_wizard
import yaml

# Local
from .datastore import (
    DATASTORES,
    MDPDatastore,
    NpyFilesDatastoreMEPS,
    init_datastore,
)


@dataclasses.dataclass
class DatastoreSelection:
    """
    Configuration for selecting a datastore to use with neural-lam.

    Attributes
    ----------
    kind : str
        The kind of datastore to use, currently `mdp` or `npyfilesmeps` are
        implemented.
    config_path : str
        The path to the configuration file for the selected datastore, this is
        assumed to be relative to the configuration file for neural-lam.
    n_boundary_points : int, optional
        Overrides :class:`MDPDatastore`'s default boundary-mask width (30),
        only valid for `kind: mdp`. Set it to 0 when the boundary forcing
        comes from a separate datastore. If unset and a separate boundary
        datastore is configured, the interior datastore uses 0.
    ds_path : str, optional
        Explicit path to the dataset (a zarr directory, or a kerchunk
        reference ``.json`` for virtual datasets), only valid for
        `kind: mdp`. If not given, the zarr next to the datastore config is
        used.
    overload_stats_path : str, optional
        Path (relative to the neural-lam config) to another mdp datastore
        config whose standardization statistics should be used instead of
        this datastore's own, only valid for `kind: mdp`.
    state_to_forcing : list of str
        State features to give to the model as forcing (taken from the
        dataset) instead of predicting them. Only supported for the interior
        mdp datastore.
    """

    kind: str
    config_path: str
    n_boundary_points: int | None = None
    ds_path: str | None = None
    overload_stats_path: str | None = None
    state_to_forcing: list[str] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        """
        Validate that the selected datastore kind is implemented.

        Raises
        ------
        ValueError
            If the provided ``kind`` is not part of :data:`DATASTORES`.
        """
        if self.kind not in DATASTORES:
            raise ValueError(f"Datastore kind {self.kind} is not implemented")


@dataclasses.dataclass
class ManualStateFeatureWeighting:
    """
    Configuration for weighting the state features in the loss function where
    the weights are manually specified.

    Attributes
    ----------
    weights : dict[str, float]
        Manual weights for the state features.
    """

    weights: dict[str, float]


@dataclasses.dataclass
class UniformFeatureWeighting:
    """
    Configuration for weighting the state features in the loss function where
    all state features are weighted equally.
    """

    pass


@dataclasses.dataclass
class OutputClamping:
    """
    Configuration for clamping the output of the model.

    Attributes
    ----------
    lower : dict[str, float]
        The minimum value to clamp each output feature to.
    upper : dict[str, float]
        The maximum value to clamp each output feature to.
    gated_features : list[str]
        State features whose predicted increment is gated by a learned
        probability of the feature exceeding a threshold (e.g. precipitation
        being non-zero). A binary cross-entropy term on the gate is added to
        the training loss.
    gate_thresholds : dict[str, float]
        Physical threshold per gated feature. Features not listed use a
        threshold of physical zero.
    """

    lower: dict[str, float] = dataclasses.field(default_factory=dict)
    upper: dict[str, float] = dataclasses.field(default_factory=dict)
    gated_features: list[str] = dataclasses.field(default_factory=list)
    gate_thresholds: dict[str, float] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class QuantileMatching:
    """
    Configuration for an additional loss term that matches the spatial
    distribution (quantiles over the grid) of the predicted state features to
    that of the target, see `metrics.sorted_mse`. The term is added to the main
    loss for every prediction step. With the `wmse` loss the term is scaled
    like the wmse term of the same feature, so that a weight of 1 makes it
    comparable to that feature's own loss term.

    Attributes
    ----------
    weights : dict[str, float]
        Weight of the quantile matching term for each state feature it should
        be applied to. State features not listed get no quantile matching term.
    """

    weights: dict[str, float] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class TrainingConfig:
    """
    Configuration related to training neural-lam

    Attributes
    ----------
    state_feature_weighting :
        ManualStateFeatureWeighting | UniformFeatureWeighting
        The method to use for weighting the state features in the loss
        function. Defaults to uniform weighting (`UniformFeatureWeighting`, i.e.
        all features are weighted equally).
    output_clamping : OutputClamping
        Per-feature lower / upper clamping bounds applied to the model output.
        Defaults to an empty ``OutputClamping`` (no clamping).
    quantile_matching : QuantileMatching
        Optional extra loss term matching the spatial value distribution of
        selected state features. Defaults to no such term.
    excluded_intervals : list[list[str]]
        Pairs of ``[start, end]`` timestamps; training samples whose time
        window overlaps any of these intervals are not used.
    """

    state_feature_weighting: (
        ManualStateFeatureWeighting | UniformFeatureWeighting
    ) = dataclasses.field(default_factory=UniformFeatureWeighting)

    output_clamping: OutputClamping = dataclasses.field(
        default_factory=OutputClamping
    )
    quantile_matching: QuantileMatching = dataclasses.field(
        default_factory=QuantileMatching
    )
    excluded_intervals: list[list[str]] = dataclasses.field(
        default_factory=list
    )


@dataclasses.dataclass
class NeuralLAMConfig(dataclass_wizard.JSONWizard, dataclass_wizard.YAMLWizard):
    """
    Configuration for the Neural-LAM model and training pipeline.

    Loads and stores all settings needed to run Neural-LAM, including
    datastore selection and training hyperparameters. Serialisation and
    deserialisation from YAML/JSON is handled via ``dataclass_wizard``.

    Attributes
    ----------
    datastores : dict[str, DatastoreSelection]
        Mapping from a user-chosen datastore name to its selection config.
        A datastore's role follows from the categories it provides: one with
        `state` data is the interior (input and output), one without is
        input-only (boundary forcing). :func:`load_config_and_datastore`
        requires exactly one interior and at most one boundary.
    training : TrainingConfig
        Configuration for training the model, including loss function and
        feature-weighting strategy. Defaults to ``TrainingConfig()``.
    """

    datastores: dict[str, DatastoreSelection]
    training: TrainingConfig = dataclasses.field(default_factory=TrainingConfig)

    class _(dataclass_wizard.JSONWizard.Meta):
        """
        Define the configuration class as a JSON wizard class.

        Together `tag_key` and `auto_assign_tags` enable that when a `Union` of
        types are used for an attribute, the specific type to deserialize to
        can be specified in the serialised data using the `tag_key` value. In
        our case we call the tag key `__config_class__` to indicate to the
        user that they should pick a dataclass describing configuration in
        neural-lam. This Union-based selection allows us to support different
        configuration attributes for different choices of methods for example
        and is used when picking between different feature weighting methods in
        the `TrainingConfig` class. `auto_assign_tags` is set to True to
        automatically set that tag key (i.e. `__config_class__` in the config
        file) should just be the class name of the dataclass to deserialize to.
        """

        tag_key = "__config_class__"
        auto_assign_tags = True
        # ensure that all parts of the loaded configuration match the
        # dataclasses used
        # TODO: this should be enabled once
        # https://github.com/rnag/dataclass-wizard/issues/137 is fixed, but
        # currently cannot be used together with `auto_assign_tags` due to a
        # bug it seems
        # raise_on_unknown_json_key = True


class InvalidConfigError(Exception):
    """Raised when the Neural-LAM configuration file is invalid or malformed."""

    pass


def load_config_and_datastore(
    config_path: str,
) -> tuple[
    NeuralLAMConfig,
    MDPDatastore | NpyFilesDatastoreMEPS,
    MDPDatastore | NpyFilesDatastoreMEPS | None,
]:
    """
    Load the neural-lam configuration and the datastores specified in the
    configuration.

    Parameters
    ----------
    config_path : str
        Path to the Neural-LAM configuration file.

    Returns
    -------
    tuple[NeuralLAMConfig, datastore, datastore_boundary]
        The configuration, the interior datastore (the one with `state`
        data), and the boundary datastore (the one without, or None).

    Raises
    ------
    InvalidConfigError
        If not exactly one datastore provides `state` data, or if more than
        one omits it.
    """
    with open(config_path, encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)
    if isinstance(raw_config, dict) and (
        "datastore" in raw_config and "datastores" not in raw_config
    ):
        raise InvalidConfigError(
            "The `datastore:` config key has been replaced by a named "
            "`datastores:` mapping, move your datastore under a named entry "
            "(`datastores: {<name>: {kind: ..., config_path: ...}}`)."
        )

    try:
        config = NeuralLAMConfig.from_yaml_file(config_path)
    except dataclass_wizard.errors.UnknownJSONKey as ex:
        raise InvalidConfigError(
            "There was an error loading the configuration file at "
            f"{config_path}. "
        ) from ex

    # Datastore config paths are relative to the neural-lam config file.
    config_dir = Path(config_path).parent
    interior_datastores = {}
    boundary_datastores = {}
    for name, selection in config.datastores.items():
        overload_stats_path = (
            None
            if selection.overload_stats_path is None
            else config_dir / selection.overload_stats_path
        )
        datastore = init_datastore(
            datastore_kind=selection.kind,
            config_path=config_dir / selection.config_path,
            n_boundary_points=selection.n_boundary_points,
            ds_path=selection.ds_path,
            overload_stats_path=overload_stats_path,
            state_to_forcing=selection.state_to_forcing or None,
        )
        if datastore.get_num_data_vars(category="state") > 0:
            interior_datastores[name] = datastore
        else:
            if selection.state_to_forcing:
                raise InvalidConfigError(
                    "state_to_forcing is only supported for the interior "
                    f"datastore, but was set for `{name}`."
                )
            boundary_datastores[name] = datastore

    if len(interior_datastores) != 1:
        raise InvalidConfigError(
            "Exactly one datastore must provide `state` data (the interior "
            f"domain), but {len(interior_datastores)} were found in "
            f"{config_path}: {sorted(interior_datastores)}."
        )
    if len(boundary_datastores) > 1:
        raise InvalidConfigError(
            "At most one boundary datastore (a datastore without `state` "
            f"data) is currently supported, but {len(boundary_datastores)} "
            f"were found in {config_path}: {sorted(boundary_datastores)}."
        )

    ((interior_name, datastore),) = interior_datastores.items()
    datastore_boundary = next(iter(boundary_datastores.values()), None)

    # With boundary forcing from a separate datastore the whole interior
    # domain is predicted, so default to no boundary mask in the interior.
    if (
        datastore_boundary is not None
        and config.datastores[interior_name].n_boundary_points is None
        and isinstance(datastore, MDPDatastore)
    ):
        datastore.set_n_boundary_points(0)

    return (
        config,
        cast("MDPDatastore | NpyFilesDatastoreMEPS", datastore),
        cast("MDPDatastore | NpyFilesDatastoreMEPS | None", datastore_boundary),
    )


@dataclasses.dataclass
class ModelConfig:
    """
    Configuration for neural-lam model architecture.

    Attributes
    ----------
    model : str, default "graph_lam"
        Model architecture name.
    graph : str, default "multiscale"
        Name of the graph to load from the datastore.
    hidden_dim : int, default 64
        Dimensionality of hidden representations.
    hidden_layers : int, default 1
        Number of hidden layers in MLPs.
    processor_layers : int, default 4
        Number of GNN layers in processor GNN.
    mesh_aggr : str, default "sum"
        Aggregation method for m2m GNN layers ("sum" or "mean").
    output_std : bool, default False
        Whether models should output standard deviation per feature.
    g2m_gnn_type : str, default "InteractionNet"
        GNN layer type for grid-to-mesh encoding.
    m2g_gnn_type : str, default "InteractionNet"
        GNN layer type for mesh-to-grid decoding.
    mesh_up_gnn_type : str, default "InteractionNet"
        GNN layer type for upward mesh message passing.
    mesh_down_gnn_type : str, default "InteractionNet"
        GNN layer type for downward mesh message passing.
    graph_dir : str or None, default None
        Directory holding the graph directories, instead of
        ``<datastore root>/graph``.
    hidden_dim_grid : int or None, default None
        Hidden dimensionality on the grid (defaults to ``hidden_dim``).
    time_delta_enc_dim : int or None, default None
        Encoding dimensionality of boundary forcing time deltas (defaults
        to ``hidden_dim``).
    shared_grid_embedder : bool, default False
        Use the same embedder for interior and boundary grid nodes.
    num_quantiles : int, default 0
        Number of quantiles to predict per output (0 for none).
    additional_graphs : list of str, default []
        Graphs of the additional training domains, see
        ``DataConfig.additional_config_paths``.
    """

    model: str = "graph_lam"
    graph: str = "multiscale"
    hidden_dim: int = 64
    hidden_layers: int = 1
    processor_layers: int = 4
    mesh_aggr: str = "sum"
    output_std: bool = False
    g2m_gnn_type: str = "InteractionNet"
    m2g_gnn_type: str = "InteractionNet"
    mesh_up_gnn_type: str = "InteractionNet"
    mesh_down_gnn_type: str = "InteractionNet"
    graph_dir: str | None = None
    hidden_dim_grid: int | None = None
    time_delta_enc_dim: int | None = None
    shared_grid_embedder: bool = False
    num_quantiles: int = 0
    additional_graphs: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class TrainRunConfig:
    """
    Hyperparameters and settings for a training or evaluation run.

    Attributes
    ----------
    epochs : int, default 200
        Number of training epochs.
    batch_size : int, default 4
        Batch size.
    ar_steps_train : int, default 1
        Autoregressive rollout steps during training.
    ar_steps_eval : int, default 10
        Autoregressive rollout steps during evaluation.
    loss : str, default "wmse"
        Loss function to use ("mse" or "wmse").
    lr : float, default 1e-3
        Learning rate.
    val_interval : int, default 1
        Validation epoch frequency.
    num_sanity_val_steps : int, default 2
        Number of validation batches to run prior to training.
    val_steps_to_log : list of int, default [1, 2, 3, 5, 10]
        Rollout steps to log during validation.
    train_steps_to_log : list of int, default []
        Rollout steps to log during training.
    metrics_watch : list of str, default []
        List of watched metrics.
    var_leads_metrics_watch : dict of {int: list of int}, default {}
        Mapping from variable index to list of watched rollout steps.
    load : str or None, default None
        Checkpoint path to load.
    restore_opt : bool, default False
        Whether to restore optimizer state from checkpoint.
    eval : str or None, default None
        Evaluation split name ("val", "test") or None if training.
    n_example_pred : int, default 1
        Number of example predictions to plot during testing.
    create_gif : bool, default False
        Whether to create animation GIFs during evaluation.
    lr_scheduler : str, default "constant"
        ``"constant"`` or ``"linear_cosine"`` (linear warm-up followed by
        cosine annealing to ``min_lr`` over ``epochs`` scheduler steps).
    min_lr : float, default 0.0
        Final learning rate of the cosine annealing.
    lr_warmup_iters : int, default 10
        Number of scheduler steps of linear warm-up.
    lr_scheduler_frequency : int, default 700
        Number of optimizer steps between scheduler steps.
    grad_checkpointing : bool, default False
        Use gradient checkpointing in-between unrolled steps.
    eval_init_times : list of int, default []
        UTC hours that validation/test forecasts are initialized at (all
        if empty).
    save_eval_to_zarr_path : str or None, default None
        Save test predictions to a zarr dataset at this path.
    plot_vars : list of str, default []
        State variables to plot example predictions for (all if empty).
    save_every_n_epochs : int or None, default None
        If set, additionally keep a checkpoint every this many epochs.
    """

    epochs: int = 200
    batch_size: int = 4
    ar_steps_train: int = 1
    ar_steps_eval: int = 10
    loss: str = "wmse"
    lr: float = 1e-3
    val_interval: int = 1
    num_sanity_val_steps: int = 2
    val_steps_to_log: list[int] = dataclasses.field(
        default_factory=lambda: [1, 2, 3, 5, 10]
    )
    train_steps_to_log: list[int] = dataclasses.field(default_factory=list)
    metrics_watch: list[str] = dataclasses.field(default_factory=list)
    var_leads_metrics_watch: dict[int, list[int]] = dataclasses.field(
        default_factory=dict
    )
    load: str | None = None
    restore_opt: bool = False
    eval: str | None = None
    n_example_pred: int = 1
    create_gif: bool = False
    lr_scheduler: str = "constant"
    min_lr: float = 0.0
    lr_warmup_iters: int = 10
    lr_scheduler_frequency: int = 700
    grad_checkpointing: bool = False
    eval_init_times: list[int] = dataclasses.field(default_factory=list)
    save_eval_to_zarr_path: str | None = None
    plot_vars: list[str] = dataclasses.field(default_factory=list)
    save_every_n_epochs: int | None = None


@dataclasses.dataclass
class DataConfig:
    """
    Data loading and forcing parameters.

    Attributes
    ----------
    num_past_forcing_steps : int, default 1
        Number of past forcing time steps as input.
    num_future_forcing_steps : int, default 1
        Number of future forcing time steps as input.
    num_past_boundary_steps : int, default 1
        Number of past boundary forcing time steps as input.
    num_future_boundary_steps : int, default 1
        Number of future boundary forcing time steps as input.
    num_workers : int, default 4
        Number of data loader workers.
    load_single_member : bool, default False
        Whether to load only ensemble member 0.
    dynamic_time_deltas : bool, default False
        Compute boundary time deltas relative to the interior state times.
    additional_config_paths : list of str, default []
        Configs of further domains to train on together with the main one.
    """

    num_past_forcing_steps: int = 1
    num_future_forcing_steps: int = 1
    num_past_boundary_steps: int = 1
    num_future_boundary_steps: int = 1
    num_workers: int = 4
    load_single_member: bool = False
    dynamic_time_deltas: bool = False
    additional_config_paths: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class ComputeConfig:
    """
    Compute and accelerator configuration.

    Attributes
    ----------
    seed : int, default 42
        Random seed.
    num_nodes : int, default 1
        Number of cluster nodes.
    devices : int or str or list of int or list of str, default ["auto"]
        Hardware devices to use.
    precision : str or int, default 32
        Floating point precision.
    """

    seed: int = 42
    num_nodes: int = 1
    devices: int | str | list[int] | list[str] = dataclasses.field(
        default_factory=lambda: ["auto"]
    )
    precision: str | int = 32


@dataclasses.dataclass
class LoggingConfig:
    """
    Experiment tracking and artifact logging configuration.

    Attributes
    ----------
    logger : str, default "wandb"
        Logger backend name ("wandb" or "mlflow").
    logger_project : str, default "neural_lam"
        Project name for the logger.
    logger_run_name : str or None, default None
        Custom run name for the logger.
    runs_root : str, default "runs"
        Root directory for run artifacts and checkpoints.
    wandb_id : str or None, default None
        W&B run ID for resuming experiments.
    log_level : str, default "INFO"
        Log level of loguru.
    """

    logger: str = "wandb"
    logger_project: str = "neural_lam"
    logger_run_name: str | None = None
    runs_root: str = "runs"
    wandb_id: str | None = None
    log_level: str = "INFO"
