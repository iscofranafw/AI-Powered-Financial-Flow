import pytest
from unittest.mock import patch,MagicMock
from services.admin_service import change_owner_status
from services.auth_service import AccessError

def test_admin_cannot_change_self():
    with patch("services.admin_service.verify_actor",return_value={"id":"same"}):
        with pytest.raises(AccessError):
            change_owner_status("token","same","aktif")

def test_admin_rejects_unknown_status():
    with patch("services.admin_service.verify_actor",return_value={"id":"admin"}):
        with pytest.raises(ValueError):
            change_owner_status("token","target","superadmin")

def test_non_owner_target_rejected():
    client=MagicMock()
    client.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value.data={"role":"kasir","status":"aktif"}
    with patch("services.admin_service.verify_actor",return_value={"id":"admin"}),patch("services.admin_service.privileged_client",return_value=client):
        with pytest.raises(AccessError):
            change_owner_status("token","target","aktif")
