# Sharing the data with DVC

DVC (Data Version Control) tracks the files under `data/` and stores the bytes in Azure Blob Storage. Git stores only a short pointer for each folder. The waves, the archives, the features, and the checkpoints stay out of GitHub.

Use the system DVC, version 3.67, which already includes the Azure remote. Do not install a second copy in the project virtual environment. If you install DVC with pip, install `dvc[azure]`. Plain `dvc` cannot talk to Azure.

`data/README.md` stays a normal Git file. Everything else under `data/` is tracked by DVC: `data/raw` (about 40 GB) and `data/cache` (about 2.3 GB).

## Commands that prepare this repo

Run these from `neural_networks_and_deep_learning_project`. They do not upload anything.

```powershell
dvc init
dvc config cache.type hardlink,copy
dvc config cache.protected true
dvc add data/raw
dvc config cache.protected false
dvc config cache.type copy
dvc add data/cache
dvc config cache.type hardlink,copy
dvc config cache.protected true
```

`dvc init` creates the `.dvc` directory. The first `dvc add` hashes the audio and hardlinks it into the local DVC cache, so the 40 GB is not copied a second time. Those audio files are then read-only. The second `dvc add` copies `data/cache`, about 2.3 GB, so a later training run can write new checkpoints without changing the hashed copy. Hashing the full 42 GB takes a while. The last two `dvc config` lines put the shared settings back: try a hardlink first, and keep checked-out files read-only.

On this machine the cache files stay writable because they were added while protection was off. On a fresh `dvc pull`, `data/cache` comes out read-only. Before training there, run `dvc unprotect data/cache`. After training changes that folder, run `dvc add data/cache` again and then `dvc push`.

Check the result before you commit:

```powershell
(Get-Item data\raw\desed\dcase_synth\audio\train\synthetic21_train\soundscapes\0.wav).IsReadOnly
(Get-Item data\cache\crnn\full\model_last.pt).IsReadOnly
git status
```

The wav should report `True`. The checkpoint should report `False`. `git status` should show `.dvc/`, `.dvcignore`, `data/.gitignore`, `data/raw.dvc`, `data/cache.dvc`, and `.gitignore`. It should not show the wav files.

Commit the pointers only. Leave the CRNN source, the CRNN reports, and the notebook edit unstaged. Do not push yet.

```powershell
git add .gitignore .dvc .dvcignore DVC.md data/.gitignore data/raw.dvc data/cache.dvc
git commit -m "Record DVC pointers for the project data."
```

## Azure, in the portal

Do this after both `dvc add` commands have finished. `dvc push` uploads the local DVC cache, so the hash step has to be done first. Sign in at https://portal.azure.com as `gaurangrgupta@outlook.com`. Use the subscription named Azure for Students. Leave the existing storage account `aiscplandgg013` alone. That one belongs to another project.

About 42 GB in Central India, Hot, locally redundant storage, is $0.02 per GB per month, about $0.85 per month. The first 100 GB downloaded each month is free.

### Resource group

1. In the top search box, type `Resource groups` and open that service.
2. Select Create.
3. Subscription: Azure for Students. If you see another subscription, switch to this one before you create anything.
4. Resource group name: `rg-nndl-desed`.
5. Region: Central India.
6. Select Review + create, then Create.
7. When the notification says the deployment is complete, select Go to resource group.

A resource group is only a folder for the bill. It does not store the audio.

### Storage account

1. Inside `rg-nndl-desed`, select Create.
2. Search the marketplace box for `Storage account` and open that result. Select Create.
3. On the Basics page:
   - Subscription: Azure for Students.
   - Resource group: `rg-nndl-desed`.
   - Storage account name: `nndldesedj013`. The portal checks the name as you type. It must be 3 to 24 characters, lowercase letters and numbers only, and unique across all of Azure. If you get a red mark, try `nndldesedj013a` or another digit. Write the name down. The later DVC commands use it.
   - Region: Central India. If Central India is not in the list, stop and do not pick a different region.
   - If you see Primary service, choose Azure Blob Storage. If you do not see that label, stay on this page.
   - Performance: Standard.
   - Redundancy: Locally-redundant storage (LRS).
4. Open the Advanced page.
   - Require secure transfer for REST API operations: checked.
   - Allow enabling anonymous access on individual containers: unchecked. The label is sometimes Allow Blob anonymous access.
   - Access tier: Hot.
   - Hierarchical namespace: unchecked. That option turns on Data Lake folders. This project uses a normal blob container.
5. Open the Networking page.
   - Network access: Enable public access from all networks. A private endpoint would block `dvc push` from home.
6. Leave Data protection, Encryption, and Tags on their defaults. Encryption should stay on Microsoft-managed keys.
7. Select Review + create, then Create. Deployment takes about a minute.
8. Select Go to resource.

### Allow the account key

DVC signs in with the account key. Some new accounts disable that key.

1. In the storage account, open Security + networking, then Access keys.
2. If the page says key access is disabled, open Settings, then Configuration. Set Allow storage account key access to Enabled and save. Go back to Access keys.
3. Select Show keys.
4. Under key1, copy Connection string. It is one line and starts with `DefaultEndpointsProtocol=https;AccountName=`.
5. Paste it into a local note, not into a file in this repo, and not into `DVC.md`. You will paste it into one PowerShell command. Anyone with this string can read and delete the container.

### Container

1. In the storage account, open Data storage, then Containers.
2. Select Container. The button may say + Container.
3. Name: `desed`.
4. Anonymous access level: Private (no anonymous access).
5. Select Create.

You do not create a folder named `dvc` in the portal. The DVC url `azure://desed/dvc` means "container `desed`, blob names starting with `dvc/`". DVC creates that prefix on the first push.

### Point DVC at the container and upload

Run this in `neural_networks_and_deep_learning_project`. If your storage account name is not `nndldesedj013`, change that one word in the second command.

```powershell
dvc remote add -d storage azure://desed/dvc
dvc remote modify storage account_name nndldesedj013
dvc remote modify --local storage connection_string "PASTE_THE_CONNECTION_STRING_HERE"
dvc push
```

The first command writes the container address into `.dvc/config` and marks it the default remote. The second command writes the account name into that same file. Neither line contains the key.

The third command writes the key into `.dvc/config.local`. The `--local` flag is what keeps the key out of Git. Keep the whole connection string inside one pair of double quotes, on one line. PowerShell will cut it if you press Enter in the middle.

`dvc push` uploads about 42 GB. The first lines can sit for a while before the percentage moves. You can stop it with Ctrl+C and run `dvc push` again. Blobs that already uploaded are skipped.

The default is `4 * cpu_count()` uploads at once. On a home connection that crowds out the large archives, and Azure closes the socket. The error looks like `Timeout on reading data from socket` and `5 files failed to upload`. The files already in the container stay there. Raise the wait and send fewer files at a time, then push again:

```powershell
dvc remote modify storage read_timeout 600
dvc remote modify storage connection_timeout 120
dvc push -j 2
```

`read_timeout 600` waits up to 10 minutes between chunks of one file. The 7 GB dcase2019 archive is the usual file that needs that. `-j 2` is only for this command. A later `dvc push -j 2` is safe and only sends what is still missing.

Check the upload in the portal: open container `desed` and select Refresh. You should see a prefix `dvc`. Opening it shows many files with long hash names. That list grows while the push is still running.

Then confirm Git did not pick up the key:

```powershell
git status
git diff .dvc/config
```

`git status` should show `.dvc/config` as modified. It should not show `.dvc/config.local`. The diff of `.dvc/config` should show the url and the account name, and it should not show `AccountKey` or `connection_string`.

```powershell
git add .dvc/config
git commit -m "Record the Azure remote for the project data."
git push
```

That push also sends the earlier local commits that were not on origin. It does not send uncommitted CRNN work.

## Collaborator

She needs GitHub access to this repo and DVC 3.67, or `dvc[azure]` if she installs with pip. She does not create a storage account, and she does not need her own credits. Her download is billed to this subscription.

Give her a shared access signature (SAS) for the one container, not the account key.

1. Open container `desed`.
2. Open Shared access tokens. On some screens it is under Settings.
3. Signing method: Account key.
4. Permissions: check Read, Add, Create, Write, Delete, and List. Leave the other boxes unchecked.
5. Start time: today. Expiry: about six months out, for example 8 April 2027. The portal uses your local clock.
6. Allowed IP addresses: leave empty, so she can pull from her own network.
7. Allowed protocols: HTTPS only.
8. Select Generate SAS token and URL.
9. Copy Blob SAS token. It starts with `sp=` or with `?sp=`. If it starts with `?`, delete that one question mark. Do not copy Blob SAS URL. That one is a full `https://` link, and DVC will not take it in the token field.
10. Send her the token and the storage account name. Do not commit either one.

On her machine, after `git pull`, she runs:

```powershell
dvc remote modify --local storage sas_token "PASTE_THE_SAS_TOKEN_HERE"
dvc pull
```

The account name and the container url are already in the committed `.dvc/config`, so she does not run `dvc remote add` again. `dvc pull` needs about 42 GB free. Her `data/cache` files come out read-only. Before she trains, she runs `dvc unprotect data/cache`. After training changes that folder, she runs `dvc add data/cache` and `dvc push`.
