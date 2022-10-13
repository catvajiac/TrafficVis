demo: reqs
  streamlit run app.py

reqs:
  pip3 install -r requirements.txt

clean:
  \rm -rf __pycache__

spotless: clean
