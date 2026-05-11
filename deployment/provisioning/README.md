# Harmonia FL Droplet Provisioning

Bootstrap Docker:

```bash
bash deployment/provisioning/bootstrap_droplet.sh
```

Upload local `.npy` data to all droplets:

```bash
DROPLETS="root@server root@client1 root@client2" bash deployment/provisioning/upload_data.sh
```

After data upload, snapshot droplets before running paid GPU verification.
