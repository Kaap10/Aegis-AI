# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY ../../requirements.txt .

# Install packages
# RUN pip install --no-cache-dir -r requirements.txt
# RUN pip install --timeout=600 --no-cache-dir -r requirements.txt
RUN pip install --timeout=600 --no-cache-dir -r requirements.txt

# Copy the dashboard code into the container
COPY . .

# Make port 8501 available
EXPOSE 8501

# Run the Streamlit app
# CMD ["streamlit", "run", "dashboard.py"]
CMD ["streamlit", "run", "src/dashboard/dashboard.py"]