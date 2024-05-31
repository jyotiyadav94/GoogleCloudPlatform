FROM pytorch/torchserve:0.7.1-cpu

ARG model_name

# install dependencies
RUN python3 -m pip install --upgrade pip
RUN pip3 install transformers==4.26.1
RUN pip3 install sentencepiece==0.1.97

USER model-server

# copy model artifacts, custom handler and other dependencies
COPY models/${model_name}/${model_name}.mar /home/model-server/model-store/model.mar

# create torchserve configuration file
USER root
RUN printf "\nservice_envelope=json" >> /home/model-server/config.properties
RUN printf "\ninference_address=http://0.0.0.0:8080" >> /home/model-server/config.properties
RUN printf "\nmanagement_address=http://0.0.0.0:8081" >> /home/model-server/config.properties
RUN printf "\njob_queue_size=100000" >> /home/model-server/config.properties
RUN printf "\ndefault_workers_per_model=1" >> /home/model-server/config.properties

USER model-server

# expose health and prediction listener ports from the image
EXPOSE 8080
EXPOSE 8081

# run Torchserve HTTP serve to respond to prediction requests
CMD ["torchserve", \
    "--start", \
    "--ts-config=/home/model-server/config.properties", \
    "--models", \
    "model=model.mar", \
    "--model-store", \
    "/home/model-server/model-store"]
