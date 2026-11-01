"""Datastore backends for loading and serving weather model data."""

# Standard library
from pathlib import Path

# Local
from .base import BaseDatastore  # noqa: F401
from .mdp import MDPDatastore  # noqa: F401
from .npyfilesmeps import NpyFilesDatastoreMEPS  # noqa: F401

DATASTORE_CLASSES = [
    MDPDatastore,
    NpyFilesDatastoreMEPS,
]

DATASTORES = {
    datastore.SHORT_NAME: datastore for datastore in DATASTORE_CLASSES
}


def init_datastore(
    datastore_kind: str,
    config_path: str | Path,
    n_boundary_points: int | None = None,
    ds_path: str | Path | None = None,
    overload_stats_path: str | Path | None = None,
    state_to_forcing: list[str] | None = None,
) -> BaseDatastore:
    """
    Instantiate a datastore based on its short-name identifier.

    Parameters
    ----------
    datastore_kind : str
        Key corresponding to one of :data:`DATASTORES`.
    config_path : str | pathlib.Path
        Path to the datastore-specific configuration file.
    n_boundary_points : int, optional
        Forwarded to :class:`~neural_lam.datastore.mdp.MDPDatastore` when
        ``datastore_kind`` is ``"mdp"``, overriding its default boundary-mask
        width. Must be left unset for other datastore kinds.
    ds_path : str | pathlib.Path, optional
        Explicit dataset path, only supported for ``"mdp"``.
    overload_stats_path : str | pathlib.Path, optional
        Config of another mdp datastore to take standardization statistics
        from, only supported for ``"mdp"``.
    state_to_forcing : list of str, optional
        State features to move to the forcing category, only supported for
        ``"mdp"``.

    Returns
    -------
    BaseDatastore
        Concrete datastore instance configured for ``config_path``.

    Raises
    ------
    NotImplementedError
        If ``datastore_kind`` is not registered, or if ``state_to_forcing``
        is set for a non-mdp datastore.
    ValueError
        If an mdp-only option is set for a ``datastore_kind`` other than
        :data:`MDPDatastore.SHORT_NAME`.
    """
    DatastoreClass = DATASTORES.get(datastore_kind)

    if DatastoreClass is None:
        raise NotImplementedError(
            f"Datastore kind {datastore_kind} is not implemented"
        )

    mdp_kwargs = {
        "n_boundary_points": n_boundary_points,
        "ds_path": ds_path,
        "overload_stats_path": overload_stats_path,
        "state_to_forcing": state_to_forcing,
    }
    mdp_kwargs = {k: v for k, v in mdp_kwargs.items() if v is not None}

    if not mdp_kwargs:
        return DatastoreClass(config_path=config_path)

    if DatastoreClass is not MDPDatastore:
        if "state_to_forcing" in mdp_kwargs:
            raise NotImplementedError(
                "state_to_forcing is only implemented for the "
                f"{MDPDatastore.SHORT_NAME} datastore"
            )
        raise ValueError(
            f"`{'`, `'.join(mdp_kwargs)}` is only supported for "
            f"`kind: {MDPDatastore.SHORT_NAME}` datastores, but was set for "
            f"a `kind: {datastore_kind}` datastore."
        )
    return MDPDatastore(config_path=config_path, **mdp_kwargs)
