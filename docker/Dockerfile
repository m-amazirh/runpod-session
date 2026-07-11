FROM lmsysorg/sglang:latest

RUN mkdir -p /models

COPY startup.sh /startup.sh
RUN chmod +x /startup.sh

EXPOSE 8080

ENTRYPOINT ["/startup.sh"]
