import pytest
from unittest.mock import patch,MagicMock
from services.auth_service import verify_actor,AccessError,register_owner

def test_pending_denied():
    admin=MagicMock()
    admin.auth.get_user.return_value.user.id="u1"
    with patch("services.auth_service.privileged_client",return_value=admin), patch("services.auth_service.own_profile",return_value={"role":"pemilik_toko","status":"pending"}):
        with pytest.raises(AccessError,match="belum aktif"):
            verify_actor("token")

def test_wrong_role_denied():
    admin=MagicMock()
    admin.auth.get_user.return_value.user.id="u1"
    with patch("services.auth_service.privileged_client",return_value=admin), patch("services.auth_service.own_profile",return_value={"role":"kasir","status":"aktif"}):
        with pytest.raises(AccessError,match="izin"):
            verify_actor("token",{"superadmin"})

def test_short_password_rejected():
    with pytest.raises(ValueError):
        register_owner("N","n@e.com","short","Shop","Bakery")
