# Local
from .base import BaseDatastore  # noqa
from .mdp import MDPDatastore  # noqa
from .npyfilesmeps import NpyFilesDatastoreMEPS  # noqa

DATASTORE_CLASSES = [
    MDPDatastore,
    NpyFilesDatastoreMEPS,
]

DATASTORES = {
    datastore.SHORT_NAME: datastore for datastore in DATASTORE_CLASSES
}


def init_datastore(
    datastore_kind,
    config_path,
    overload_stats_path=None,
    ds_path=None,
    state_to_forcing=None,
):
    DatastoreClass = DATASTORES.get(datastore_kind)

    if DatastoreClass is None:
        raise NotImplementedError(
            f"Datastore kind {datastore_kind} is not implemented"
        )

    kwargs = {}
    if state_to_forcing:
        if DatastoreClass is not MDPDatastore:
            raise NotImplementedError(
                "state_to_forcing is only implemented for the "
                f"{MDPDatastore.SHORT_NAME} datastore"
            )
        kwargs["state_to_forcing"] = state_to_forcing

    datastore = DatastoreClass(
        config_path=config_path,
        overload_stats_path=overload_stats_path,
        ds_path=ds_path,
        **kwargs,
    )

    return datastore
