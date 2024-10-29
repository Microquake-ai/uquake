from google.protobuf.timestamp_pb2 import Timestamp

def convert_utc_to_grpc_timestamp(utc_datetime):
    # Create a Timestamp object
    grpc_timestamp = Timestamp()

    # Set seconds and nanoseconds based on UTCDateTime
    grpc_timestamp.seconds = int(utc_datetime.timestamp)  # Convert to int for seconds
    grpc_timestamp.nanos = int((utc_datetime.timestamp - grpc_timestamp.seconds) * 1e9)

    return grpc_timestamp
