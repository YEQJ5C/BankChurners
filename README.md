# BankChurners Projekt - MLOps Féléves Feladat

## 1. Fejlesztesi Kornyezet Kialakitasa (Environment Setup)

A projekt egy dedikalt virtualizalt kornyezetben lett kifejlesztve es tesztelve a maximalis reprodukalhatosag erdekeben.

### Virtualis Gep es Operacios Rendszer
- Virtualizacio: VMware Workstation 17 Player
- OS: Ubuntu 22.04.5 LTS (Jammy Jellyfish)
  - Distributor ID: Ubuntu
  - Description: Ubuntu 22.04.5 LTS
  - Release: 22.04
  - Codename: jammy
- Integracio: `sudo apt install open-vm-tools-desktop -y` (biztositja a vagolap-megosztast es egerintegraciot).

### Python es Virtualis Kornyezet
- Alap Python verzio: `Python 3.10.12` (`/usr/bin/python3`)
- Pip csomagkezelo: `sudo apt install -y python3-pip` (`pip 22.0.2`)
- Virtuális környezet: `sudo apt install -y python3-venv` (`/usr/lib/python3.10/venv/__init__.py`, Version: `3.10.6-1~22.04.1`)

### Verziokovetes (Git)
- Git verzio: `Git 2.34.1`
- Globalis konfiguracio:
  - `user.name`: `YEQJ5C`
  - `user.email`: `YEQJ5C@stud.uni-obuda.hu`

### Kontenerizacio (Docker es Docker Compose)
A Docker a hivatalos tárolóból lett telepítve a legfrissebb verzió biztosításához:
1. Függőségek és GPG kulcs beállítása:
   ```bash
   sudo apt update && sudo apt install -y ca-certificates curl gnupg
   sudo install -m 0755 -d /etc/apt/keyrings
   sudo curl -fsSL [https://download.docker.com/linux/ubuntu/gpg](https://download.docker.com/linux/ubuntu/gpg) -o /etc/apt/keyrings/docker.asc
   sudo chmod a+r /etc/apt/keyrings/docker.asc

   