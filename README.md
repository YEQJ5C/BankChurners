# BankChurners Projet - MLOps Féléves Feladat

## 1. Fejlesztesi Kornyezet Kialakitasa (Environment Setup)

A projekt egy dedikalt virtualizalt kornyezetben lett kifejlesztve es tesztelve.

### Virtualis Gep es Operacios Rendszer
* Virtualizacio: VMware Workstation 17 Player
* OS: Ubuntu 22.04.5 LTS (Jammy Jellyfish)
    * LSB modules: No LSB modules are available.
    * Distributor ID: Ubuntu
    * Description: Ubuntu 22.04.5 LTS
    * Release: 22.04
    * Codename: jammy
* VM Eszkozok: sudo apt install open-vm-tools-desktop -y (biztositja a gazda-vendeg vagolap-megosztast).

### Python es Virtualis Kornyezet
* Alap Python verzio: Python 3.10.12 (/usr/bin/python3)
* Csomagkezelo: python3-pip (pip 22.0.2 from /usr/lib/python3/dist-packages/pip)
* Virtualis kornyezet: python3-venv (/usr/lib/python3.10/venv/__init__.py, Version: 3.10.6-1~22.04.1)

### Verziokovetes (Git)
* Git verzio: Git 2.34.1
* Globalis konfiguracio:
  * user.name: YEQJ5C
  * user.email: YEQJ5C@stud.uni-obuda.hu

### Kontenerizacio (Docker)
* Docker / Docker Compose: Elokeszitve az docker.io csomaggal (version 29.1.3-0ubuntu3~22.04.2) a konteneres futtatashoz.