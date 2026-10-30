import hashlib
from unittest.mock import patch,MagicMock
from services.cashier_service import create_cashier


def test_cashier_role_is_server_controlled():
    client=MagicMock()
    client.auth.admin.create_user.return_value.user.id='new-cashier-uuid'
    with patch('services.cashier_service.verify_actor',return_value={
        'id':'owner-uuid','role':'pemilik_toko','toko_id':'shop123'}),\
         patch('services.cashier_service.privileged_client',return_value=client),\
         patch('services.cashier_service.secrets.token_urlsafe',return_value='unguessable-token'):
        create_cashier('token','Kasir','kasir@example.com','password123')
    invite=client.table.return_value.insert.call_args.args[0]
    assert invite['token_hash']==hashlib.sha256(b'unguessable-token').hexdigest()
    assert invite['created_by']=='owner-uuid'
    payload=client.auth.admin.create_user.call_args.args[0]
    assert payload['user_metadata']['bizflow_cashier_ticket']=='unguessable-token'
    assert 'app_metadata' not in payload
    assert client.auth.admin.update_user_by_id.call_args.args[1]['app_metadata']['bizflow_role']=='kasir'
