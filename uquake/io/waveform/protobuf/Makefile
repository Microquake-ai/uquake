.PHONY: generate_proto

generate_proto:
	@echo "Generating Python gRPC files from proto..."
	@python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. uquake.proto
	@echo "Generation complete."
