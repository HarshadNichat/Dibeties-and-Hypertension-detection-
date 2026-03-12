# Project Commands

This file contains the main commands required to create, configure, and run the project from the beginning.

## Windows PowerShell Commands

### 1. Go to the project folder

```powershell
Set-Location "D:\dibeties prediction project\Dibetes_and_hypertension_project"
```

### 2. Create the virtual environment

```powershell
python -m venv .venv
```

### 3. Activate the virtual environment

```powershell
& ".venv\Scripts\Activate.ps1"
```

### 4. Upgrade pip

```powershell
python -m pip install --upgrade pip
```

### 5. Install all dependencies

```powershell
pip install -r requirements.txt
```

### 6. Run the Streamlit application

```powershell
python -m streamlit run app.py
```

### 7. Optional: run the app with the exact virtual environment python

```powershell
& ".venv\Scripts\python.exe" -m streamlit run app.py
```

### 8. Optional: validate syntax before running

```powershell
python -m py_compile app.py
```

### 9. Optional: install one new package manually

```powershell
pip install reportlab
```

### 10. Stop the Streamlit app

In the terminal where Streamlit is running, press:

```text
Ctrl + C
```
