# Runs hang_debug.py against the code from just before the Network handling commit (d52c9c7).
$env:PYTHONPATH = "C:\Users\hd01\Documents\GitHub\ImageLynx_before_view\src"
& "C:\Users\hd01\Documents\GitHub\ImageLynx\.venv\Scripts\python.exe" -c "import haemolynx, sys; print('using', haemolynx.__file__)"
& "C:\Users\hd01\Documents\GitHub\ImageLynx\.venv\Scripts\python.exe" "C:\Users\hd01\Documents\GitHub\ImageLynx\hang_debug.py"
