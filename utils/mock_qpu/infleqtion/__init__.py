# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

############################################################################################
# OPTION 1:
############################################################################################

# from fastapi import FastAPI, HTTPException, Header
# from typing import Union
# import uvicorn
# import uuid
# from pydantic import BaseModel

# app = FastAPI()


# # Job submission model
# class JobSubmission(BaseModel):
#     qasm_strs: list
#     target: str
#     shots: int
#     method: str


# # Job status model
# class JobStatus(BaseModel):
#     status: str
#     samples: Union[dict, None] = None


# # Dictionary to store job statuses and results
# jobs_db = {}


# @app.post("/v0.2.0/jobs")
# async def submit_job(job: JobSubmission, authorization: Union[str, None] = Header(default=None)):
#     if authorization is None:
#         raise HTTPException(status_code=401, detail="Authorization header missing")

#     # Generate a unique job ID
#     job_id = str(uuid.uuid4())

#     # Store the job with initial status "Running"
#     jobs_db[job_id] = {
#         "status": "Running",
#         "shots": job.shots,
#         "qasm_strs": job.qasm_strs,
#         "method": job.method,
#         "samples": None,  # Will be filled when the job is done
#     }

#     # Return the job IDs in the response
#     return {"job_ids": [job_id]}


# @app.get("/v0.2.0/job/{job_id}")
# async def get_job_status(job_id: str, authorization: Union[str, None] = Header(default=None)):
#     if authorization is None:
#         raise HTTPException(status_code=401, detail="Authorization header missing")

#     if job_id not in jobs_db:
#         raise HTTPException(status_code=404, detail="Job not found")

#     job = jobs_db[job_id]

#     # Simulate job processing: after a certain number of checks, mark the job as "Done"
#     if job["status"] == "Running":
#         # For simplicity, set the status to "Done" immediately
#         job["status"] = "Done"

#         # Generate some dummy samples
#         from random import randint

#         num_qubits = 5  # Assume 5 qubits for this example
#         num_samples = job["shots"]

#         samples = {}
#         for _ in range(num_samples):
#             bitstring = "".join(str(randint(0, 1)) for _ in range(num_qubits))
#             samples[bitstring] = samples.get(bitstring, 0) + 1

#         job["samples"] = samples

#     # Prepare the response
#     response = {"status": job["status"]}

#     if job["status"] == "Done":
#         response["samples"] = job["samples"]

#     return response

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)

############################################################################################
# OPTION 2:
############################################################################################

# import uvicorn
# from fastapi import FastAPI, HTTPException, Header
# from typing import Union
# from pydantic import BaseModel
# import uuid
# import asyncio

# # Define the REST Server App
# app = FastAPI()

# # In-memory storage for jobs and their statuses
# jobs = {}
# job_results = {}
# processing_time = 5  # seconds to simulate job processing


# # Data model for job submission
# class JobSubmission(BaseModel):
#     qasm_strs: list
#     target: str
#     shots: int
#     method: Union[str, None] = None  # Optional field


# # Endpoint to submit a job
# @app.post("/v0.2.0/jobs")
# async def submit_job(job: JobSubmission, authorization: Union[str, None] = Header(default=None)):
#     if authorization is None:
#         raise HTTPException(status_code=401, detail="Authorization header missing")

#     # Generate a unique job ID
#     job_id = str(uuid.uuid4())
#     jobs[job_id] = {"status": "Queued", "submission": job, "result": None}

#     # Simulate job processing in the background
#     asyncio.create_task(process_job(job_id))

#     # Return the job ID in the expected format
#     response = {"job_ids": [job_id]}
#     return response


# # Function to simulate job processing
# async def process_job(job_id):
#     # Update job status to 'Running'
#     jobs[job_id]["status"] = "Running"

#     # Simulate processing time
#     await asyncio.sleep(processing_time)

#     # Generate mock results
#     job = jobs[job_id]["submission"]
#     shots = job.shots
#     qasm_strs = job.qasm_strs
#     # For simplicity, we'll generate random counts for '0' and '1'

#     # Mock sample results
#     from random import randint

#     counts = {}
#     for _ in range(shots):
#         bitstring = "".join(
#             ["0" if randint(0, 1) == 0 else "1" for _ in range(1)]
#         )  # Adjust qubit count if needed
#         counts[bitstring] = counts.get(bitstring, 0) + 1

#     # Update job status and results
#     jobs[job_id]["status"] = "Done"
#     jobs[job_id]["result"] = {"samples": counts}


# # Endpoint to get job status and results
# @app.get("/v0.2.0/job/{job_id}")
# async def get_job_status(job_id: str, authorization: Union[str, None] = Header(default=None)):
#     if authorization is None:
#         raise HTTPException(status_code=401, detail="Authorization header missing")

#     if job_id not in jobs:
#         raise HTTPException(status_code=404, detail="Job ID not found")

#     job = jobs[job_id]
#     status = job["status"]
#     response = {"status": status}

#     # Include samples if the job is done
#     if status == "Done":
#         response["samples"] = job["result"]["samples"]

#     return response


# def start_server():
#     uvicorn.run(app, port=8000, host="0.0.0.0", log_level="info")


# if __name__ == "__main__":
#     start_server()
