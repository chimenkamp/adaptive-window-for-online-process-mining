# Event Stream Window Estimation

## Install and run Kafka

### RUN Zookeeper 

```bash
zkserver start
```

### RUN Kafka
```bash
kafka-server-start /opt/homebrew/etc/kafka/server.properties
```

### Create and test Topic "completeness_estimation"
```bash
kafka-topics --create --topic completenessEstimation --bootstrap-server localhost:9092 --replication-factor 1 --partitions 1
```