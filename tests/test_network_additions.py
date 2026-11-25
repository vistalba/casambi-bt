"""Tests for applying network updates that only add units, groups or scenes."""

import copy
import json
import pathlib
from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from CasambiBt import Casambi, NetworkAdditions
from CasambiBt._cache import Cache
from CasambiBt._network import Network, _NetworkSession

UNIT_TYPE = {
    "id": 10,
    "model": "Model",
    "vendor": "Vendor",
    "mode": "Mode",
    "stateLength": 1,
    "controls": [],
}


def _unit(deviceId: int, name: str | None = None) -> dict[str, Any]:
    return {
        "type": 10,
        "deviceID": deviceId,
        "uuid": f"uuid-{deviceId}",
        "address": f"addr-{deviceId}",
        "name": name or f"Unit {deviceId}",
        "firmware": "1.0",
    }


def _group(groupId: int, units: list[int]) -> dict[str, Any]:
    return {
        "type": 2,
        "groupID": groupId,
        "name": f"Group {groupId}",
        "cells": [{"type": 1, "unit": u} for u in units],
    }


def _network(
    revision: int,
    units: list[dict[str, Any]],
    groups: list[dict[str, Any]] | None = None,
    scenes: list[int] | None = None,
) -> dict[str, Any]:
    return {
        "status": "UPDATED",
        "network": {
            "revision": revision,
            "name": "My Network",
            "protocolVersion": 10,
            "keyStore": {
                "keys": [{"id": 1, "type": 1, "role": 3, "name": "k", "key": "00" * 16}]
            },
            "units": units,
            "grid": {"cells": groups or []},
            "scenes": [{"sceneID": s, "name": f"Scene {s}"} for s in (scenes or [])],
        },
    }


def _response(network: dict[str, Any]) -> httpx.Response:
    return httpx.Response(200, json=network, content=json.dumps(network).encode())


@pytest.fixture
async def cache(tmp_path: pathlib.Path) -> AsyncGenerator[Cache, None]:
    c = Cache(tmp_path)
    await c.setUuid("test-uuid")
    await c.invalidateCache()
    yield c


@pytest.fixture
def mock_client() -> AsyncMock:
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get.return_value = httpx.Response(200, json=UNIT_TYPE)
    return client


@pytest.fixture
async def network(mock_client: AsyncMock, cache: Cache) -> Network:
    network = Network("test-uuid", mock_client, cache)
    await network.load()
    network._session = _NetworkSession(
        "sess", "net", True, 1, datetime.now(UTC) + timedelta(days=1)
    )
    network._id = "test-net"
    mock_client.put.return_value = _response(
        _network(1, [_unit(1), _unit(2)], [_group(1, [1])], [1])
    )
    await network.update()
    return network


async def _download(
    network: Network, mock_client: AsyncMock, update: dict[str, Any]
) -> None:
    mock_client.put.return_value = _response(update)
    assert await network.fetchUpdate()


async def test_additions_are_applied(network: Network, mock_client: AsyncMock):
    """New units, groups and scenes are added, existing objects stay."""
    unit1, unit2 = network.units
    group1 = network.groups[0]
    scene1 = network.scenes[0]

    await _download(
        network,
        mock_client,
        _network(
            2,
            [_unit(1), _unit(2), _unit(3)],
            [_group(1, [1]), _group(2, [2, 3])],
            [1, 2],
        ),
    )
    additions = await network.applyAdditions()

    assert additions is not None
    assert [u.deviceId for u in additions.units] == [3]
    assert [g.groudId for g in additions.groups] == [2]
    assert [s.sceneId for s in additions.scenes] == [2]
    # The objects in use are kept, so references to them stay valid.
    assert network.units[:2] == [unit1, unit2]
    assert network.units[0] is unit1
    assert network.groups[0] is group1
    assert network.scenes[0] is scene1
    # A new group references the units in use, not new copies of them.
    assert network.groups[1].units[0] is unit2
    assert network.groups[1].units[1] is additions.units[0]
    assert network.revision == 2


async def test_nothing_to_apply(network: Network):
    """Without a newer configuration nothing is added."""
    assert await network.applyAdditions() == NetworkAdditions([], [], [])


async def test_applied_configuration_is_updated(
    network: Network, mock_client: AsyncMock
):
    """A second update is compared against the configuration applied last."""
    await _download(
        network,
        mock_client,
        _network(2, [_unit(1), _unit(2), _unit(3)], [_group(1, [1])], [1]),
    )
    assert await network.applyAdditions() is not None

    # Unit 3 is part of the applied configuration now, so removing it again
    # is a change that can't be applied live.
    await _download(
        network, mock_client, _network(3, [_unit(1), _unit(2)], [_group(1, [1])], [1])
    )

    assert await network.applyAdditions() is None
    assert [u.deviceId for u in network.units] == [1, 2, 3]


@pytest.mark.parametrize(
    "update",
    [
        pytest.param(_network(2, [_unit(1)], [_group(1, [1])], [1]), id="unit removed"),
        pytest.param(
            _network(2, [_unit(1, "Renamed"), _unit(2)], [_group(1, [1])], [1]),
            id="unit renamed",
        ),
        pytest.param(_network(2, [_unit(1), _unit(2)], [], [1]), id="group removed"),
        pytest.param(
            _network(2, [_unit(1), _unit(2)], [_group(1, [1, 2])], [1]),
            id="group members changed",
        ),
        pytest.param(
            _network(2, [_unit(1), _unit(2)], [_group(1, [1])], []), id="scene removed"
        ),
    ],
)
async def test_other_changes_are_not_applied(
    network: Network, mock_client: AsyncMock, update: dict[str, Any]
):
    """Anything but additions is left for the next connect."""
    await _download(network, mock_client, update)

    assert await network.applyAdditions() is None
    assert [u.deviceId for u in network.units] == [1, 2]
    assert network.units[0].name == "Unit 1"
    assert network.revision == 1


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("name", "Other"),
        ("protocolVersion", 11),
        ("keyStore", {"keys": []}),
        ("managerKey", "11" * 16),
    ],
)
async def test_changed_network_settings_are_not_applied(
    network: Network, mock_client: AsyncMock, key: str, value: Any
):
    """New keys or settings need a new connection."""
    update = _network(2, [_unit(1), _unit(2), _unit(3)], [_group(1, [1])], [1])
    update["network"][key] = value
    await _download(network, mock_client, update)

    assert await network.applyAdditions() is None
    assert [u.deviceId for u in network.units] == [1, 2]


async def test_scene_renamed_is_not_applied(network: Network, mock_client: AsyncMock):
    update = _network(2, [_unit(1), _unit(2)], [_group(1, [1])], [1])
    update["network"]["scenes"][0]["name"] = "Renamed"
    await _download(network, mock_client, update)

    assert await network.applyAdditions() is None


async def test_malformed_update_is_not_applied(
    network: Network, mock_client: AsyncMock
):
    update = _network(2, [_unit(1), _unit(2)], [_group(1, [1])], [1])
    del update["network"]["grid"]
    await _download(network, mock_client, update)

    assert await network.applyAdditions() is None
    assert len(network.units) == 2


async def test_no_applied_configuration(mock_client: AsyncMock, cache: Cache):
    """A network that was never loaded has nothing to compare with."""
    network = Network("test-uuid", mock_client, cache)

    assert await network.applyAdditions() is None


async def test_casambi_delegates_to_the_network():
    casa = Casambi(MagicMock())
    network = MagicMock(spec=Network)
    network._networkRevision = 1
    additions = NetworkAdditions([], [], [])
    network.applyAdditions = AsyncMock(return_value=additions)
    casa._casaNetwork = network

    assert await casa.applyNetworkAdditions() is additions


async def test_backup_after_additions_contains_them(
    network: Network, mock_client: AsyncMock
):
    """The applied configuration is the one in the cache."""
    update = _network(2, [_unit(1), _unit(2), _unit(3)], [_group(1, [1])], [1])
    await _download(network, mock_client, copy.deepcopy(update))
    assert await network.applyAdditions() is not None

    cached = await network._readCachedNetwork()

    assert cached is not None
    assert cached["network"]["revision"] == 2
    assert len(cached["network"]["units"]) == 3
