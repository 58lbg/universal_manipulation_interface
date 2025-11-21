apt-get install libopencv-dev libopencv-contrib-dev
git clone https://github.com/ceres-solver/ceres-solver
cd ceres-solver
git checkout 2.1.0
mkdir -p build && cd build && cmake .. -DBUILD_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release
make -j install

cd ../../

git clone https://github.com/urbste/pyTheiaSfM
cd pyTheiaSfM && git checkout 69c3d37 && mkdir -p build && cd build
cmake .. && make -j
make install

cd ../../

mkdir -p build && cd build && cmake ..
make -j

cd ../

conda activate umi
pip install -r requirements.txt