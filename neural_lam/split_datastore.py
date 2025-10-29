"""
Splitting a datastore into two new datastores that contain the interiour domain and its boundary.
"""

# Standard library
import argparse
import copy
from datetime import datetime, timedelta

# Third-party
import mllam_data_prep as mdp
from loguru import logger


def main():
    """
    Main function to split a datastore into interior and boundary datastores.
    """

    parser = argparse.ArgumentParser(
        description="Split a datastore into interior and boundary datastores."
    )
    parser.add_argument(
        "--source_config_path",
        type=str,
        required=True,
        help="Path to the datastore configuration file.",
    )
    parser.add_argument(
        "--interior_config_path",
        type=str,
        required=True,
        help="Path to save the interior datastore configuration file.",
    )
    parser.add_argument(
        "--boundary_config_path",
        type=str,
        required=True,
        help="Path to save the boundary datastore configuration file.",
    )

    args = parser.parse_args()

    datastore_config_path = args.source_config_path
    datastore_interior_config_path = args.interior_config_path
    datastore_boundary_config_path = args.boundary_config_path

    cfg = mdp.Config.from_yaml_file(datastore_config_path)
    domain_cropping = copy.deepcopy(cfg.output.domain_cropping)
    output_splitting = copy.deepcopy(cfg.output.splitting)

    start_time_str = str(
        output_splitting.splits["train"].start
    )  # Example start time as string
    time_interval = timedelta(minutes=10)  # Example interval duration

    start_time = datetime.fromisoformat(start_time_str)

    for split in cfg.output.splitting.splits:
        cfg.output.splitting.splits[split].start = start_time.isoformat()
        cfg.output.splitting.splits[split].end = (
            start_time + time_interval
        ).isoformat()
        cfg.output.splitting.splits[split].compute_statistics = None

    cfg.output.domain_cropping = None
    logger.info("Creating temporary datastore for convex hull calculation.")
    datastore = mdp.create_dataset(config=cfg)

    max_dist = domain_cropping.margin_width_degrees
    logger.info(
        f"Calculating convex hull with margin width of {max_dist} degrees."
    )
    ds_boundary, mask = mdp.ops.cropping.crop_with_convex_hull(
        ds=datastore,
        ds_reference=datastore,
        margin_thickness=max_dist,
        include_interior_points=True,
        return_mask=True,  # invert=True
    )
    logger.info("Get interior and boundary indices.")
    interior_ind = datastore.grid_index.values[~mask]
    boundary_ind = ds_boundary.grid_index.values
    logger.info("Set up interior and boundary datastores.")
    cfg_interiour = copy.deepcopy(cfg)
    cfg_boundary = copy.deepcopy(cfg)

    for ipt in cfg_interiour.inputs.values():
        ipt.coord_ranges["cell"] = list(interior_ind)

    for ipt in cfg_boundary.inputs.values():
        ipt.coord_ranges["cell"] = list(
            set(boundary_ind).union(set(interior_ind))
        )
        if ipt.target_output_variable == "state":
            ipt.target_output_variable = "forcing"
            ipt.dim_mapping["forcing_feature"] = ipt.dim_mapping[
                "state_feature"
            ]
            _ = ipt.dim_mapping.pop("state_feature", None)
    cfg_interiour.output.domain_cropping = None
    cfg_interiour.output.splitting = output_splitting

    cfg_boundary.output.domain_cropping = {
        "interior_dataset_config_path": datastore_interior_config_path,
        "include-interior-points": False,
        "margin-width-degrees": -1 * domain_cropping.margin_width_degrees,
    }
    cfg_boundary.output.variables["forcing"] = [
        "time",
        "grid_index",
        "forcing_feature",
    ]
    _, cfg_boundary.output.variables.pop("state", None)
    cfg_boundary.output.splitting = output_splitting
    logger.info("Saving the interior and boundary datastores.")
    cfg_boundary.to_yaml_file(datastore_boundary_config_path)
    cfg_interiour.to_yaml_file(datastore_interior_config_path)


if __name__ == "__main__":
    main()
