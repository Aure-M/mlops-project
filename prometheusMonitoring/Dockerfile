FROM prom/prometheus:v2.30.0

WORKDIR /etc/prometheus

ADD prometheus.yml /etc/prometheus/prometheus.yml

EXPOSE 9090

ENTRYPOINT ["prometheus", "--config.file=/etc/prometheus/prometheus.yml"]
