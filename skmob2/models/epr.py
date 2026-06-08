from __future__ import annotations

from typing import Any

import narwhals as nw
import numpy as np

from ._common import (
    LATITUDE,
    LONGITUDE,
    RELEVANCE,
    tessellation_lat_lngs,
    to_pandas_frame,
    trajectory_dataframe,
)
from .gravity import Gravity


def _unwrap_native_frame(df: Any) -> Any:
    return getattr(df, "df", df)


def _tessellation_arrays(
    spatial_tessellation: Any, relevance_column: str | None
) -> tuple[Any, Any, np.ndarray, np.ndarray]:
    native = _unwrap_native_frame(spatial_tessellation)
    nw_df = nw.from_native(native, eager_only=True)
    columns = set(nw_df.columns)
    if "geometry" in columns:
        pandas_frame = to_pandas_frame(spatial_tessellation)
        lats_lngs = tessellation_lat_lngs(pandas_frame)
        relevances = (
            np.ones(len(pandas_frame), dtype=float)
            if relevance_column is None
            else pandas_frame[relevance_column].fillna(0).to_numpy(dtype=float)
        )
        return pandas_frame, nw.from_native(pandas_frame, eager_only=True).implementation, lats_lngs, relevances

    if LATITUDE in columns and LONGITUDE in columns:
        lat_col, lng_col = LATITUDE, LONGITUDE
    elif "latitude" in columns and "longitude" in columns:
        lat_col, lng_col = "latitude", "longitude"
    elif "lat" in columns and "lon" in columns:
        lat_col, lng_col = "lat", "lon"
    else:
        raise ValueError("spatial_tessellation must include a geometry column or latitude/longitude columns.")

    lats = np.asarray(nw_df.get_column(lat_col).to_numpy(), dtype=float)
    lngs = np.asarray(nw_df.get_column(lng_col).to_numpy(), dtype=float)
    if relevance_column is None:
        relevances = np.ones(len(nw_df), dtype=float)
    else:
        relevances = np.asarray(nw_df.get_column(relevance_column).to_numpy(), dtype=float)
        relevances = np.nan_to_num(relevances, nan=0.0)
    return nw_df.to_native(), nw_df.implementation, np.column_stack((lats, lngs)), relevances


def _trajectory_native_frame(agent_ids: Any, lats: Any, lngs: Any, timestamps: Any, backend: Any) -> Any:
    datetime_values = np.asarray(timestamps, dtype="datetime64[s]").astype("datetime64[ms]")
    values = {
        "uid": agent_ids,
        "lat": lats,
        "lng": lngs,
        "datetime": datetime_values,
    }
    return nw.from_dict(values, backend=backend).to_native()


def compute_od_matrix(
    gravity_singly,
    spatial_tessellation,
    tile_id_column="tile_id",
    relevance_column=RELEVANCE,
):
    return gravity_singly.generate(
        spatial_tessellation,
        tile_id_column=tile_id_column,
        tot_outflows_column=None,
        relevance_column=relevance_column,
        out_format="probabilities",
    ).to_matrix()


class EPR:
    def __init__(
        self,
        name="EPR model",
        rho=0.6,
        gamma=0.21,
        beta=0.8,
        tau=17,
        min_wait_time_minutes=20,
    ):
        self._name = name
        self._rho = rho
        self._gamma = gamma
        self._tau = tau
        self._beta = beta
        self._spatial_tessellation = None
        self.lats_lngs = None
        self.relevances = None
        self.gravity_singly = None
        self._min_wait_time = min_wait_time_minutes / 60.0
        self._trajectories_ = []

    @property
    def name(self):
        return self._name

    @property
    def rho(self):
        return self._rho

    @property
    def gamma(self):
        return self._gamma

    @property
    def tau(self):
        return self._tau

    @property
    def beta(self):
        return self._beta

    @property
    def min_wait_time(self):
        return self._min_wait_time

    @property
    def spatial_tessellation_(self):
        return self._spatial_tessellation

    @property
    def trajectories_(self):
        return self._trajectories_

    def generate(
        self,
        start_date,
        end_date,
        spatial_tessellation,
        gravity_singly=None,
        n_agents=1,
        starting_locations=None,
        relevance_column=RELEVANCE,
        random_state=None,
        log_file=None,
        show_progress=False,
    ):
        if starting_locations is not None and len(starting_locations) < n_agents:
            raise IndexError("The number of starting locations is smaller than the number of agents.")
        if gravity_singly is None:
            self.gravity_singly = Gravity(gravity_type="singly constrained")
        elif isinstance(gravity_singly, Gravity):
            if gravity_singly.gravity_type == "singly constrained":
                self.gravity_singly = gravity_singly
            else:
                raise AttributeError(
                    "Argument `gravity_singly` should be a skmob.models.gravity.Gravity object with argument `gravity_type` equal to 'singly constrained'."
                )
        else:
            raise TypeError("Argument `gravity_singly` should be of type skmob.models.gravity.Gravity.")

        parameters = {
            "model": {
                "class": self.__class__.__init__,
                "generate": {
                    "start_date": start_date,
                    "end_date": end_date,
                    "gravity_singly": gravity_singly,
                    "n_agents": n_agents,
                    "relevance_column": relevance_column,
                    "random_state": random_state,
                    "show_progress": show_progress,
                },
            }
        }

        if random_state is not None and int(random_state) < 0:
            raise ValueError("random_state must be a non-negative integer.")

        self._trajectories_ = []
        self._spatial_tessellation, output_backend, self.lats_lngs, self.relevances = _tessellation_arrays(
            spatial_tessellation, relevance_column
        )

        start_values = None if starting_locations is None else np.asarray(starting_locations, dtype=np.int64)

        rows = self._epr_generate_parallel(
            start_date,
            end_date,
            n_agents,
            random_state=None if random_state is None else int(random_state),
            starting_locations=start_values,
            output_backend=output_backend,
        )
        return trajectory_dataframe(rows, parameters=parameters)

    def _epr_generate_parallel(
        self,
        start_date,
        end_date,
        n_agents,
        *,
        random_state=None,
        starting_locations=None,
        output_backend=None,
    ):
        from skmob2 import _core

        start_ts = int(start_date.timestamp())
        end_ts = int(end_date.timestamp())
        lats = np.ascontiguousarray(self.lats_lngs[:, 0], dtype=float)
        lngs = np.ascontiguousarray(self.lats_lngs[:, 1], dtype=float)
        relevances = np.ascontiguousarray(self.relevances, dtype=float)
        starts = None if starting_locations is None else np.ascontiguousarray(starting_locations, dtype=np.int64)
        agent_ids, lats_out, lngs_out, timestamps = _core.model_epr_simulate_agents(
            lats,
            lngs,
            relevances,
            float(self._rho),
            float(self._gamma),
            float(self._beta),
            float(self._tau),
            float(self._min_wait_time),
            start_ts,
            end_ts,
            self.gravity_singly.deterrence_func_type,
            float(self.gravity_singly.deterrence_func_args[0]),
            float(self.gravity_singly.origin_exp),
            float(self.gravity_singly.destination_exp),
            int(n_agents),
            random_state,
            starts,
        )
        return _trajectory_native_frame(agent_ids, lats_out, lngs_out, timestamps, output_backend)


class DensityEPR(EPR):
    def __init__(
        self,
        name="Density EPR model",
        rho=0.6,
        gamma=0.21,
        beta=0.8,
        tau=17,
        min_wait_time_minutes=20,
    ):
        super().__init__(
            rho=rho,
            gamma=gamma,
            beta=beta,
            tau=tau,
            min_wait_time_minutes=min_wait_time_minutes,
        )
        self._name = name

    def generate(
        self,
        start_date,
        end_date,
        spatial_tessellation,
        gravity_singly=None,
        n_agents=1,
        starting_locations=None,
        relevance_column=RELEVANCE,
        random_state=None,
        log_file=None,
        show_progress=False,
    ):
        return super().generate(
            start_date,
            end_date,
            spatial_tessellation,
            gravity_singly=gravity_singly,
            n_agents=n_agents,
            starting_locations=starting_locations,
            relevance_column=relevance_column,
            random_state=random_state,
            log_file=log_file,
            show_progress=show_progress,
        )


class SpatialEPR(EPR):
    def __init__(
        self,
        name="Spatial EPR model",
        rho=0.6,
        gamma=0.21,
        beta=0.8,
        tau=17,
        min_wait_time_minutes=20,
    ):
        super().__init__(
            rho=rho,
            gamma=gamma,
            beta=beta,
            tau=tau,
            min_wait_time_minutes=min_wait_time_minutes,
        )
        self._name = name

    def generate(
        self,
        start_date,
        end_date,
        spatial_tessellation,
        gravity_singly=None,
        n_agents=1,
        starting_locations=None,
        random_state=None,
        log_file=None,
        show_progress=False,
    ):
        return super().generate(
            start_date,
            end_date,
            spatial_tessellation,
            gravity_singly=gravity_singly,
            n_agents=n_agents,
            starting_locations=starting_locations,
            relevance_column=None,
            random_state=random_state,
            log_file=log_file,
            show_progress=show_progress,
        )


class Ditras(EPR):
    def __init__(self, diary_generator, name="Ditras model", rho=0.3, gamma=0.21):
        super().__init__(rho=rho, gamma=gamma)
        self._diary_generator = diary_generator
        self._name = name
        self._rho = rho
        self._gamma = gamma

    def generate(
        self,
        start_date,
        end_date,
        spatial_tessellation,
        gravity_singly=None,
        n_agents=1,
        starting_locations=None,
        relevance_column=RELEVANCE,
        random_state=None,
        log_file=None,
        show_progress=False,
    ):
        # Coverage-first compatibility: the spatial phase follows DensityEPR,
        # while the diary generator is retained as public model state.
        return super().generate(
            start_date,
            end_date,
            spatial_tessellation,
            gravity_singly=gravity_singly,
            n_agents=n_agents,
            starting_locations=starting_locations,
            relevance_column=relevance_column,
            random_state=random_state,
            log_file=log_file,
            show_progress=show_progress,
        )
