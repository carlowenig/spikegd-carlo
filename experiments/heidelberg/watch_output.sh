#!/bin/bash


JOB_LIST=$(squeue --me -h -o "%A:%j:%T")

if [ -z "$JOB_LIST" ]; then
    echo "No running SLURM jobs found."
    exit 1
fi

get_output_file() {
    job_id=$1
    # Get the output file path from the job script, assuming the job uses "--output" option
    output_file=$(scontrol show job "$job_id" | grep -oP 'StdOut=\K\S+')
    if [ "$output_file" = "STDOUT" ]; then
        # Default output file path if not explicitly specified
        output_file="slurm-${job_id}.out"
    fi
    echo "$output_file"
}

# If only one job is running, select it automatically
if [ $(echo "$JOB_LIST" | wc -l) -eq 1 ]; then
    JOB_ID=$(echo "$JOB_LIST" | cut -d':' -f1)
    JOB_NAME=$(echo "$JOB_LIST" | cut -d':' -f2)
    JOB_STATE=$(echo "$JOB_LIST" | cut -d':' -f3)

    echo "Automatically selected the only job in queue: $JOB_NAME ($JOB_ID, $JOB_STATE)"
else
    # Present a selection menu if multiple jobs are running
    echo "Select a job to watch its output:"
    select JOB in $(echo "$JOB_LIST" | sed 's/:/ /'); do
        JOB_ID=$(echo "$JOB" | cut -d' ' -f1)
        JOB_NAME=$(echo "$JOB" | cut -d' ' -f2)
        JOB_STATE=$(echo "$JOB" | cut -d' ' -f3)
        break
    done
fi

echo "Waiting for job to start. Currently in state $JOB_STATE."
while [ $JOB_STATE != "RUNNING" ]
do
    sleep 2
    JOB_STATE=$(squeue -j "$JOB_ID" -h -o "%T")
done

OUTPUT_FILE=$(get_output_file "$JOB_ID")

# Loop until the output file is found
echo "Waiting for output file: $OUTPUT_FILE"
while [ ! -f "$OUTPUT_FILE" ]
do
    # echo "Output file $OUTPUT_FILE not found, waiting..."
    sleep 2  # Check every 2 seconds for the output file
done

# Once the file is found, start tailing it
echo "Tailing output file: $OUTPUT_FILE"
echo "============================================="
# cat "$OUTPUT_FILE"
tail -f -n +1 "$OUTPUT_FILE"