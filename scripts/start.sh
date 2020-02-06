cd "$(dirname "${BASH_SOURCE[0]}")"
cd ..

DOCKER_MAJOR_VERSION_STRING=$(docker -v | grep -oP '([0-9]+)' | sed -n 1p)
DOCKER_MINOR_VERSION_STRING=$(docker -v | grep -oP '([0-9]+)' | sed -n 2p)
ps_test=$(docker ps -a)
datasets=("assin-ptpt" "assin-ptbr")
DOCKER_MAJOR_VERSION=$((10#$DOCKER_MAJOR_VERSION_STRING))
DOCKER_MINOR_VERSION=$((10#$DOCKER_MINOR_VERSION_STRING))
BERT_DIR=$BERT_DIR

if [[ ! -z $ps_test ]]; then
    build_test=$(docker image ls | grep 'ruanchaves/elmo.*2.0')
else
    build_test=$(sudo docker image ls | grep 'ruanchaves/elmo.*2.0')
fi

if [[ -z $build_test ]] && [[ ! -z $ps_test ]]; then 
    docker build -t ruanchaves/elmo:2.0 .
elif [[ -z $build_test ]] && [[ -z $ps_test ]]; then
    sudo docker build -t ruanchaves/elmo:2.0 .
fi

if [[ ! -z $ps_test ]] && [[ $DOCKER_MAJOR_VERSION -ge 19 ]] && [[ $DOCKER_MINOR_VERSION -ge 3 ]]; then
    docker run --gpus all \
        -v `pwd`:/home \
        --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
        --env BERT_DIR=$BERT_DIR \
        -it --rm ruanchaves/assin:2.0 bash -c "source /home/scripts/run_assin.sh"
elif [[ ! -z $ps_test ]] && [[ $DOCKER_MAJOR_VERSION -le 19 ]] && [[ $DOCKER_MINOR_VERSION -le 2 ]]; then
    nvidia-docker run \
        -v `pwd`:/home \
        --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
        --env BERT_DIR=$BERT_DIR \
        -it --rm ruanchaves/assin:2.0 bash . /home/scripts/run_assin.sh
elif [[ -z $ps_test ]] && [[ $DOCKER_MAJOR_VERSION -ge 19 ]] && [[ $DOCKER_MINOR_VERSION -ge 3 ]]; then
    sudo -E docker run --gpus all \
        -v `pwd`:/home \
        --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
        --env BERT_DIR=$BERT_DIR \
        -it --rm ruanchaves/assin:2.0 bash . /home/scripts/run_assin.sh
elif [[ -z $ps_test ]] && [[ $DOCKER_MAJOR_VERSION -le 19 ]] && [[ $DOCKER_MINOR_VERSION -le 2 ]]; then
    sudo -E nvidia-docker run \
        -v `pwd`:/home \
        --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
        --env BERT_DIR=$BERT_DIR \
        -it --rm ruanchaves/assin:2.0 bash -c "source /home/scripts/run_assin.sh"
fi