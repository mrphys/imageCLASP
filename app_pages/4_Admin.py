import streamlit as st

password = st.text_input("Enter a password", type="password")

accessed = False
if password == 'admin':
    accessed = True


if accessed:
    st.write('You have admin access')