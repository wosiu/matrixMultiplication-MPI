#include <iostream>
#include <string>
#include <fstream>      // std::ifstream
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cassert>
#include <getopt.h>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <utility>      // std::move (objects)

#include <chrono> // debug only
#include <thread> // debug only

#include <mpi.h>

#include "densematgen.h"

#define checkpointon false
#define debon false
#define BR {MPI_Finalize(); return 0;}
#define SEQ {this_thread::sleep_for (std::chrono::seconds(mpi_rank)); }
#define CP {if (checkpointon) cout << endl << "CHECKPOINT@" << __LINE__ << endl;}
#define deb(burak) if(debon) {cout<<__LINE__<< " DEB-> "<<#burak<<": "<<burak<<endl;}
#define debv(burak) if(debon) {cout<<__LINE__<< " DEB-> "<<#burak<<": \t"; for(unsigned int zyx=0;zyx<burak.size();zyx++) cout<<burak[zyx]<<" "; cout<<endl;}
#define debt(burak,SIzE) if(debon) {cout<<__LINE__<< " DEB-> "<<#burak<<": \t"; for(int zyx=0;zyx<(int)SIzE;zyx++) cout<<burak[zyx]<<" "; cout<<endl;}
#define debend if(debon) {cout<<"_____________________"<<endl;}

using namespace std;

struct MatrixInfo {
	int base, add, nrow, ncol, start_row, start_col;
};

class Matrix {
public:
	int nrow = 0, ncol = 0;
};

typedef double* dense_type;

class DenseMatrix: public Matrix {
private:

public:
	dense_type cells = nullptr; // [col][row] = col*nrow+row
	int start_col = 0;
	int start_row = 0;
	int last_col_excl = 0;
	int last_row_excl = 0;

	void init(int nrow, int ncol, int start_row, int start_col) {
		if (this->cells != nullptr) {
			delete[] this->cells;
		}
		this->cells = new double[nrow * ncol](); // continous block of memory

		this->nrow = nrow;
		this->ncol = ncol;
		this->start_row = start_row;
		this->start_col = start_col;
		this->last_row_excl = start_row + nrow;
		this->last_col_excl = start_col + ncol;
	}

	DenseMatrix() :
			Matrix() {
	}

	// Initialize matrix with 0
	DenseMatrix(int nrow, int ncol, int start_row = 0, int start_col = 0) :
			Matrix() {
		init(nrow, ncol, start_row, start_col);
	}

	DenseMatrix(MatrixInfo other) :
			DenseMatrix(other.nrow, other.ncol, other.start_row, other.start_col) {
	}

	// Initialize only dimensions, does not copy values
	DenseMatrix(const DenseMatrix& other) :
			DenseMatrix(other.nrow, other.ncol, other.start_row, other.start_col) {
	}

	~DenseMatrix() {
		if (cells != nullptr) {
			delete[] cells;
		}
	}

	friend void swap(DenseMatrix& a, DenseMatrix& b) {
		using std::swap;
		// bring in swap for built-in types
		swap(a.cells, b.cells);

		swap(a.nrow, b.nrow);
		swap(a.ncol, b.ncol);
		swap(a.start_row, b.start_row);
		swap(a.start_col, b.start_col);
		swap(a.last_row_excl, b.last_row_excl);
		swap(a.last_col_excl, b.last_col_excl);
	}

	void setCell(int global_x, int global_y, double val) const {
		global_x -= this->start_row;
		global_y -= this->start_col;
//		return cells[global_x][global_y];
		cells[global_x + global_y * nrow] = val;
	}

	dense_type data() {
		return cells;
	}

	bool containsCell(int global_x, int global_y) const {
		return global_x >= start_row && global_y >= start_col && global_x < last_row_excl && global_y < last_col_excl;
	}

	// unsafe
	double getCell(int global_x, int global_y) const {
		global_x -= this->start_row;
		global_y -= this->start_col;
//		return cells[global_x][global_y];
		return cells[global_x + global_y * nrow];
	}

	void add(int global_x, int global_y, double v) {
		global_x -= this->start_row;
		global_y -= this->start_col;
		cells[global_x + global_y * nrow] += v;
	}

	bool isEmpty() const {
		return nrow == 0 || ncol == 0;
	}

	void setCells(int v = 0) {
		for (int i = 0; i < nrow * ncol; i++) {
			cells[i] = v;
		}
	}

	void print() const {
		printf("%d %d\n", nrow, ncol);
		for (int x = 0; x < nrow; x++) {
			for (int y = 0; y < ncol; y++) {
				printf(" %lf", cells[x + nrow * y]);
			}
			if (x < nrow - 1) {
				printf("\n");
			}
		}
	}
};

/**
 * x - rows, y - columns
 */
struct cell {
	int x, y;
	double v;
};

typedef vector<cell> sparse_type;

class SparseMatrix: public Matrix {
public:
	sparse_type cells;

	void print() const {
		for (cell ce : cells) {
			printf("%d, %d: %lf\n", ce.x, ce.y, ce.v);
		}
	}

	bool isEmpty() const {
		return cells.empty();
	}

	// for debug only
	operator DenseMatrix() const {
		DenseMatrix casted(nrow, ncol, 0, 0);
		for (auto& cell : cells) {
			casted.setCell(cell.x, cell.y, cell.v);
		}
		return casted;
	}

};

// for debug only
ostream & operator <<(std::ostream & s, const cell & v) {
	return s << '(' << v.x << ',' << v.y << ":" << v.v << ")";
}

void partialMul(DenseMatrix& C, const SparseMatrix& A, const DenseMatrix& B) {
	for (auto& sparse : A.cells) {
		// could avoid starting from start_col by scaling sparse.y -= start.col,, sparse.x -= start.row
		for (int k = B.start_col; k < B.last_col_excl; k++) {
			const double v = sparse.v * B.getCell(sparse.y, k);
			C.add(sparse.x, k, v);
		}
	}
}

MatrixInfo blocked1DSubMatrixInfo(int rank, int num_processes, int dim) {
	MatrixInfo info;
	if (rank >= num_processes) {
		throw invalid_argument(
				"Given rank bigger eq than number of processes: " + to_string(rank) + ">=" + to_string(num_processes));
	}
	info.base = dim / num_processes; // base number of columns for each process
	info.add = dim % num_processes; // number of processes with +1 column than the rest
	info.nrow = dim;
	info.ncol = info.base + int(rank < info.add);
	info.start_row = 0;
	info.start_col = rank * info.base + min(rank, info.add);
	return info;
}

// only for debug purpose
DenseMatrix generateWholeDense(int N, int seed) {
	DenseMatrix matrix(N, N, 0, 0);

	for (int y = 0; y < N; y++) {
		for (int x = 0; x < N; x++) {
			//matrix.cells[x][y] = generate_double(seed, x + start_row, y + start_col);
			matrix.setCell(x, y, generate_double(seed, x, y));
		}
	}
	return matrix;
}

DenseMatrix generateDenseSubmatrix(int mpi_rank, int num_processes, int N, int seed) {
	auto info = blocked1DSubMatrixInfo(mpi_rank, num_processes, N);
	DenseMatrix matrix(info);

	for (int y = info.start_col; y < info.start_col + info.ncol; y++) {
		for (int x = info.start_row; x < info.start_row + info.nrow; x++) {
			//matrix.cells[x][y] = generate_double(seed, x + start_row, y + start_col);
			matrix.setCell(x, y, generate_double(seed, x, y));
		}
	}
	return matrix;
}

SparseMatrix readCSR(ifstream & stream) {
	ios::sync_with_stdio(false);
	int row_num, col_num, nz_num, max_num_nz_row;

	if (!(stream >> row_num >> col_num >> nz_num >> max_num_nz_row).good()) {
		throw runtime_error("Wrong format of sparse matrix - header");
	}
	vector<cell> cells(nz_num);

	for (int i = 0; i < nz_num; i++) {
		if (!(stream >> cells[i].v).good()) {
			throw runtime_error("Wrong format of sparse matrix - values");
		}
	}

	int cell_i = 0;
	int offset;

	if (!(stream >> offset).good()) { // omit
		throw runtime_error("Wrong format of sparse matrix - offset omit");
	}
	for (int row_i = 0; row_i < row_num; row_i++) {
		if (!(stream >> offset).good()) {
			throw runtime_error("Wrong format of sparse matrix - offset");
		}
		for (; cell_i < offset; cell_i++) {
			cells[cell_i].x = row_i;
		}
	}

// JA
	for (int i = 0; i < nz_num; i++) {
		if (!(stream >> cells[i].y).good()) {
			throw runtime_error("Wrong format of sparse matrix - columns");
		}
	}

	char ch = 0;
	if (!stream.eof()) {
		stream >> ch;
	}
	if (ch != 0 || !stream.eof()) {
		throw runtime_error("Wrong format of sparse matrix - some data unread, check if no EOL at the end of file.");
	}

	SparseMatrix matrix;
	matrix.cells = cells;
	matrix.nrow = row_num;
	matrix.ncol = col_num;
	return matrix;
}

/**
 * Historical - unused in the final solution.
 * Did not want to work with debugger.
 */
SparseMatrix readCSR(FILE * stream) {
	int row_num, col_num, nz_num, max_num_nz_row;
	if (fscanf(stream, "%d %d %d %d ", &row_num, &col_num, &nz_num, &max_num_nz_row) != 4) {
		throw runtime_error("Wrong format of sparse matrix - header");
	}
	vector<cell> cells(nz_num);

	for (int i = 0; i < nz_num; i++) {
		if (fscanf(stream, "%lf ", &cells[i].v) != 1) {
			throw runtime_error("Wrong format of sparse matrix - values");
		}
	}

	int cell_i = 0;
	int offset;
	if (fscanf(stream, "%d ", &offset) != 1) { // omit
		throw runtime_error("Wrong format of sparse matrix - offset omit");
	}
	for (int row_i = 0; row_i < row_num; row_i++) {
		if (fscanf(stream, "%d ", &offset) != 1) {
			throw runtime_error("Wrong format of sparse matrix - offset");
		}
		for (; cell_i < offset; cell_i++) {
			cells[cell_i].x = row_i;
		}
	}

// JA
	for (int i = 0; i < nz_num; i++) {
		if (fscanf(stream, "%d ", &cells[i].y) != 1) {
			throw runtime_error("Wrong format of sparse matrix - columns");
		}
	}

	if (!feof(stream)) {
		throw runtime_error("Wrong format of sparse matrix - some data unread.");
	}
	SparseMatrix matrix;
	matrix.cells = cells;
	matrix.nrow = row_num;
	matrix.ncol = col_num;
	return matrix;
}

SparseMatrix readCSR(char* path) {
//	FILE * pFile = fopen(optarg, "r");
//	if (pFile == NULL) {
//		throw runtime_error("Cannot open sparse matrix file " + path);
//	}
//	SparseMatrix sparse = readCSR(pFile);
//	fclose(pFile);

	ifstream txtFile;
	txtFile.open(path, ifstream::in);
	if (!txtFile.is_open()) {
		throw runtime_error("Cannot open sparse matrix file " + string(path));
	}
	SparseMatrix sparse = readCSR(txtFile);
	txtFile.close();
	return sparse;
}

void chunkSparse(SparseMatrix& sparse, int num_processes, int sparse_mode, int* scounts, int* displs) {

	if (sparse_mode <= 1) {
		if (sparse_mode == 1) {
			// sort by cell value
			sort(sparse.cells.begin(), sparse.cells.end(), [](const cell&a, const cell& b) {return a.v < b.v;});
		}
		int nz = sparse.cells.size();
		int base = nz / num_processes; // base number of columns for each process
		int add = nz % num_processes; // number of processes with +1 cell than the rest

		for (int i = 0; i < num_processes; i++) {
			scounts[i] = base + int(i < add);
			displs[i] = (i == 0) ? 0 : displs[i - 1] + scounts[i - 1];
		}
	} else if (sparse_mode == 2) {
		// sort by col
		sort(sparse.cells.begin(), sparse.cells.end(), [](const cell&a, const cell& b) {return a.y < b.y;});
		int n = sparse.ncol;
		// assign cells by equal distribution of columns to each process
		unsigned int cel_it = 0;
		for (int p = 0; p < num_processes; p++) {
			auto info = blocked1DSubMatrixInfo(p, num_processes, n);
			scounts[p] = 0;
			displs[p] = cel_it;
			int last_col_excl = info.start_col + info.ncol;
			// need to find out where column start in continuous cells vector
			// could be some tricky bound with own cmp in log..
			for (; sparse.cells[cel_it].y < last_col_excl && cel_it < sparse.cells.size(); cel_it++) {
				scounts[p]++;
			}
		}
	} else if (sparse_mode == 3) {
		// sort by row
		sort(sparse.cells.begin(), sparse.cells.end(), [](const cell&a, const cell& b) {return a.x < b.x;});
		int n = sparse.nrow;
		// assign cells by equal distribution of columns to each process
		unsigned int cel_it = 0;
		for (int p = 0; p < num_processes; p++) {
			auto info = blocked1DSubMatrixInfo(p, num_processes, n);
			scounts[p] = 0;
			displs[p] = cel_it;
			// this is no mistake that i calculate last_row using *col fields
			// as function above generates info for column blocked, so we "transpose"
			int last_row_excl = info.start_col + info.ncol;
			// could be some tricky bound with own cmp in log..
			for (; sparse.cells[cel_it].x < last_row_excl && cel_it < sparse.cells.size(); cel_it++) {
				scounts[p]++;
			}
		}
	} else {
		throw invalid_argument("Wrong sparse mode: " + to_string(sparse_mode));
	}
}

MPI_Datatype commitSparseCellType() {
	const int nitems = 3;
	int blocklengths[nitems] = { 1, 1, 1 };
	MPI_Datatype types[nitems] = { MPI_INT, MPI_INT, MPI_DOUBLE };
	MPI_Datatype mpi_type;
	MPI_Aint offsets[nitems];

	offsets[0] = offsetof(cell, x);
	offsets[1] = offsetof(cell, y);
	offsets[2] = offsetof(cell, v);

	MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_type);
	MPI_Type_commit(&mpi_type);
	return mpi_type;
}

sparse_type sparseScatterV(SparseMatrix& sparse, MPI_Datatype MPI_SPARSE_CELL, int num_processes, int mpi_rank,
		int sparse_mode) {
	sparse_type my_sparse_chunk;

	cell* sendbuf = NULL;
	int scounts[num_processes];
	int displs[num_processes];

	if ((mpi_rank) == 0) {
		// may sort cells in some order regarding sparse_mode
		chunkSparse(sparse, num_processes, sparse_mode, scounts, displs);
		sendbuf = sparse.cells.data();
	}

// scatter chunks sizes
	int mysize = -1;
	MPI_Scatter(scounts, 1, MPI_INT, &mysize, 1, MPI_INT, 0, MPI_COMM_WORLD);

	my_sparse_chunk.resize(mysize);
// scatter data
	MPI_Scatterv(sendbuf, scounts, displs, MPI_SPARSE_CELL, my_sparse_chunk.data(), mysize, MPI_SPARSE_CELL, 0,
	MPI_COMM_WORLD);

	return my_sparse_chunk;
}

int main(int argc, char * argv[]) {

	bool use_inner = false;
	int show_results = 0, gen_seed = -1, repl_fact = 1, option = -1, exponent = 1, count_ge = 0, sparse_mode = 0,
			sparse_mode_default = 1, print_stat = 0;
	int num_processes = 1, mpi_rank = 0;
	double comm_start = 0, comm_end = 0, comp_start = 0, comp_end = 0;
	double ge_element = 0;
	SparseMatrix sparse;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
// init MPI
	auto MPI_SPARSE_CELL = commitSparseCellType();

	while ((option = getopt(argc, argv, "vixs:f:c:e:g:m:")) != -1) {
		switch (option) {
			case 'v':
				show_results = 1;
				break;
			case 'i':
				use_inner = true;
				break;
			case 'f':
				if ((mpi_rank) == 0) {
					try {
						sparse = readCSR(optarg);
					} catch (runtime_error const& e) {
						fprintf(stderr, "%s : %s", e.what(), optarg);
						MPI_Finalize();
						return 3;
					}
				}
				break;
			case 'c':
				repl_fact = atoi(optarg);
				break;
			case 's':
				gen_seed = atoi(optarg);
				break;
			case 'e':
				exponent = atoi(optarg);
				break;
			case 'g':
				count_ge = 1;
				ge_element = atof(optarg);
				break;
			case 'm':
				sparse_mode_default = 0;
				sparse_mode = atoi(optarg);
				break;
			case 'x':
				print_stat = 1;
				break;
			default:
				fprintf(stderr, "error parsing argument %c exiting\n", option);
				MPI_Finalize();
				return 3;
		}
	}
	auto repl_fact_sq = repl_fact * repl_fact;

// VALIDATE OPTIONS

	if ((gen_seed == -1) || ((mpi_rank == 0) && sparse.isEmpty())) {
		fprintf(stderr, "error: missing seed or sparse matrix file; exiting\n");
		MPI_Finalize();
		return 3;
	}

	if (sparse_mode_default == 1) {
		if (use_inner) {
			sparse_mode = 3;
		} else {
			sparse_mode = 0;
		}
	}

	if (use_inner) {
		if (num_processes % repl_fact_sq != 0) {
			fprintf(stderr, "error: squared replication factor should divide number of processes; exiting\n");
			MPI_Finalize();
			return 3;
		}
		if (sparse_mode != 3) {
			sparse_mode = 3;
			fprintf(stderr, "Inner algorithm require row-splitting of sparse matrix, mode omitted.\n");
		}
	}

	if (!use_inner) {
		if (num_processes % repl_fact != 0) {
			fprintf(stderr, "error: replication factor should divide number of processes; exiting\n");
			MPI_Finalize();
			return 3;
		}

	}
	CP;

// START COMMUNICATION - replicate matrixes
// REPLICATE SPARSE (the same for both Alg)
	comm_start = MPI_Wtime();

// broadcast matrix dimension
	int N = (mpi_rank == 0) ? sparse.nrow : -1;
	MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

// get sparse chunk
	sparse_type my_sparse_chunk;
	my_sparse_chunk = sparseScatterV(sparse, MPI_SPARSE_CELL, num_processes, mpi_rank, sparse_mode);

	CP;
// create replication group
	MPI_Comm MPI_COMM_REPL;
	MPI_Request repl_sparse_req;
	MPI_Comm_split(MPI_COMM_WORLD, mpi_rank / repl_fact, mpi_rank, &MPI_COMM_REPL);

// replicate within group
	int my_sparse_size = my_sparse_chunk.size();

// exchange with everyone within replication group number of owned cells
	int chunks_sizes[repl_fact], chunks_displs[repl_fact];
	MPI_Allgather(&my_sparse_size, 1, MPI_INT, chunks_sizes, 1, MPI_INT, MPI_COMM_REPL);
	CP

	int repl_sparse_size = chunks_sizes[0]; //incremented below
	chunks_displs[0] = 0;
	for (int i = 1; i < repl_fact; i++) {
		chunks_displs[i] = chunks_displs[i - 1] + chunks_sizes[i - 1];
		repl_sparse_size += chunks_sizes[i];
	}

// gather all chunks within replication group
	sparse_type repl_sparse_chunk;
	repl_sparse_chunk.resize(repl_sparse_size);
	CP

		// waits for finish following after generating dense matrix
	MPI_Iallgatherv(my_sparse_chunk.data(), my_sparse_size, MPI_SPARSE_CELL, repl_sparse_chunk.data(), chunks_sizes,
			chunks_displs, MPI_SPARSE_CELL, MPI_COMM_REPL, &repl_sparse_req);

	if ((debon)) {
		printf("Communication %d: %.5f\n", mpi_rank, comm_end - comm_start);
	}

// REPPLICATE DENSE (only for inner)

	DenseMatrix my_dense;
	CP;

	if (use_inner) {
		DenseMatrix my_dense_chunk = generateDenseSubmatrix(mpi_rank, num_processes, N, gen_seed);

		auto my_dense_size = my_dense_chunk.nrow * my_dense_chunk.ncol; // total number of cells

		int chunks_sizes[repl_fact], chunks_displs[repl_fact];
		int repl_rank_start = (mpi_rank / repl_fact) * repl_fact;
		auto fstInfo = blocked1DSubMatrixInfo(repl_rank_start, num_processes, N);
		chunks_sizes[0] = N * fstInfo.ncol;
		chunks_displs[0] = 0;
		auto repl_dense_ncol = fstInfo.ncol;

		for (int i = 1; i < repl_fact; i++) {
			auto proc = i + repl_rank_start;
			auto info = blocked1DSubMatrixInfo(proc, num_processes, N);
			chunks_sizes[i] = info.ncol * N;
			chunks_displs[i] = chunks_displs[i - 1] + chunks_sizes[i - 1];
			repl_dense_ncol += info.ncol;
		}
		my_dense.init(N, repl_dense_ncol, 0, fstInfo.start_col);

		// blocking gather - there is no stuff to do meanwhile anyway
		CP;

		MPI_Allgatherv(my_dense_chunk.cells, my_dense_size, MPI_DOUBLE, my_dense.cells, chunks_sizes, chunks_displs,
		MPI_DOUBLE, MPI_COMM_REPL);

		CP;
	} else {
		DenseMatrix tmp = generateDenseSubmatrix(mpi_rank, num_processes, N, gen_seed);
		swap(my_dense, tmp);
	}

	CP;

	DenseMatrix partial_res(my_dense); // initialize with 0
	SparseMatrix my_sparse;
	MPI_Wait(&repl_sparse_req, MPI_STATUS_IGNORE);
	my_sparse.cells = std::move(repl_sparse_chunk);


	comm_end = MPI_Wtime();
	comp_start = MPI_Wtime();


// COMPUTE RESULT MATRIX

	int rounds_num = -1;
	MPI_Request reqs[2];
	MPI_Status stats[2];
	int incoming_size;
	sparse_type sparse_buff;

// ALG 2 PREPARE (INNER)

	if (use_inner) {
		// SHIFT SPARSE LAYERS

		int q = num_processes / repl_fact_sq;
		rounds_num = q;
		int l = mpi_rank % repl_fact;

		// if (l > 0)
		int shift = l * q * repl_fact;
		int next_proc = (mpi_rank + shift) % num_processes;
		int prev_proc = (mpi_rank - shift + num_processes) % num_processes;
//		SEQ deb(mpi_rank) deb(shift) deb(next_proc) deb(prev_proc) deb(q) deb(l)

		CP
		MPI_Isend(my_sparse.cells.data(), my_sparse.cells.size(), MPI_SPARSE_CELL, next_proc, 0, MPI_COMM_WORLD,
				reqs + 0);

		CP

		MPI_Probe(prev_proc, 0, MPI_COMM_WORLD, stats + 0);
		CP
		MPI_Get_count(stats + 0, MPI_SPARSE_CELL, &incoming_size);
		sparse_buff.resize(incoming_size);
		CP

		MPI_Irecv(sparse_buff.data(), incoming_size, MPI_SPARSE_CELL, prev_proc, 0, MPI_COMM_WORLD, reqs + 1);

		CP
		MPI_Waitall(2, reqs, stats);

		CP
		for (int st = 0; st < 2; st++) {
			if (stats[st].MPI_ERROR != MPI_SUCCESS) {
				throw runtime_error("Error in non-blocking send/receive: " + to_string(st));
			}
		}

		swap(my_sparse.cells, sparse_buff); // O(1)
		sparse_buff.clear();

	} else {
		rounds_num = num_processes / repl_fact;
	}

	CP;

// COMPUTATIONS ALG 1 + 2 (commnon)

	int next_proc = (mpi_rank + repl_fact) % num_processes;
	int prev_proc = (mpi_rank - repl_fact + num_processes) % num_processes;

	for (int exp_i = 0; exp_i < exponent; ++exp_i) {
		partial_res.setCells(0);

		for (int r = 0; r < rounds_num; ++r) {
			MPI_Isend(my_sparse.cells.data(), my_sparse.cells.size(), MPI_SPARSE_CELL, next_proc, 0, MPI_COMM_WORLD,
					reqs + 0);
			CP
			partialMul(partial_res, my_sparse, my_dense); // does not modify my_sparse

			MPI_Probe(prev_proc, 0, MPI_COMM_WORLD, stats + 0);
			CP
			MPI_Get_count(stats + 0, MPI_SPARSE_CELL, &incoming_size);
			sparse_buff.resize(incoming_size);
			CP
			MPI_Irecv(sparse_buff.data(), incoming_size, MPI_SPARSE_CELL, prev_proc, 0, MPI_COMM_WORLD, reqs + 1);
			CP
			MPI_Waitall(2, reqs, stats);

			for (int st = 0; st < 2; st++) {
				if (stats[st].MPI_ERROR != MPI_SUCCESS) {
					throw runtime_error("Error in non-blocking send/receive: " + to_string(st));
				}
			}

			swap(my_sparse.cells, sparse_buff); // O(1)
			sparse_buff.clear();
		}

		if (use_inner) {
			// exchange sub-matrixes between processes in replication group
			CP
			MPI_Allreduce(partial_res.cells, my_dense.cells, partial_res.nrow * partial_res.ncol, MPI_DOUBLE,
			MPI_SUM, MPI_COMM_REPL);

		} else {
			swap(my_dense.cells, partial_res.cells); // O(1)
		}
	}
	comp_end = MPI_Wtime();
	CP
	if ( debon) {
		printf("Computations %d: %.5f\n", mpi_rank, comp_end - comp_start);
	}

// PRINT RESULTS

	MPI_Comm MPI_COMM_REPRESENTATIVES;
	bool is_representative = (mpi_rank % repl_fact) == 0;
	MPI_Comm_split(MPI_COMM_WORLD, mpi_rank % repl_fact, mpi_rank, &MPI_COMM_REPRESENTATIVES);
	MPI_Comm MPI_COMM_RESULTS;

	CP;

	if (use_inner) {
		MPI_COMM_RESULTS = MPI_COMM_REPRESENTATIVES;
	} else {
		MPI_COMM_RESULTS = MPI_COMM_WORLD;
	}

//agreg_src_proc.shrink_to_fit();
	bool do_agregation = (!use_inner) || ((use_inner) && is_representative);

	CP;
	if (show_results && do_agregation) {
		// gather results to root
		int s = (mpi_rank == 0) ? N : 0;
		int* rcounts = nullptr;
		int* displs = nullptr;
		double* result = nullptr;

		if (mpi_rank == 0) {
			result = new double[s * s]; // on heap, will be wrapped and freed by DanseMatrix destructor
			// iterate through representatives
			if (use_inner) {
				int agreg_src_proc_num = num_processes / repl_fact;
				rcounts = new int[agreg_src_proc_num];
				displs = new int[agreg_src_proc_num];

				for (int i = 0; i < agreg_src_proc_num; ++i) {
					int repr_proc = i * repl_fact;
					rcounts[i] = 0;
					for (int proc_i = repr_proc; proc_i < repl_fact + repr_proc; proc_i++) {
						// for inner gets dimensions of whole replication group
						auto info = blocked1DSubMatrixInfo(proc_i, num_processes, N);
						rcounts[i] += info.ncol * info.nrow;
					}
					displs[i] = (i == 0) ? 0 : displs[i - 1] + rcounts[i - 1];
				}

			} else {
				rcounts = new int[num_processes];
				displs = new int[num_processes];

				for (int i = 0; i < num_processes; ++i) {
					// for inner gets dimensions of whole replication group
					auto info = blocked1DSubMatrixInfo(i, num_processes, N);
					rcounts[i] = info.ncol * info.nrow; /* note change from previous example */
					displs[i] = (i == 0) ? 0 : displs[i - 1] + rcounts[i - 1];
				}

			}
		}
		CP;

		// gather only from representatives
		MPI_Gatherv(my_dense.cells, my_dense.nrow * my_dense.ncol, MPI_DOUBLE, result, rcounts, displs,
		MPI_DOUBLE, 0, MPI_COMM_RESULTS);

		if (rcounts != nullptr) {
			delete[] rcounts;
			delete[] displs;
		}

		CP;
		if (mpi_rank == 0) {
			if (debon) {
				printf("SPARSE:\n");
				DenseMatrix(sparse).print();
				printf("\nDENSE:\n");
				DenseMatrix dense_whole = generateWholeDense(N, gen_seed);
				dense_whole.print();
				DenseMatrix noParRes(dense_whole);
				partialMul(noParRes, sparse, dense_whole);
				printf("\nRESULT no parallel:\n");
				noParRes.print();
				printf("\nRESULT:\n");
			}

			DenseMatrix resultM;
			resultM.cells = result; // will free result
			resultM.nrow = N;
			resultM.ncol = N;
			resultM.print();
		}
		CP;
	}

	if (count_ge && do_agregation) {
		int my_counter = 0, final_count;
		for (int i = 0; i < my_dense.nrow * my_dense.ncol; i++) {
			my_counter += int(my_dense.cells[i] >= ge_element);
		}
		MPI_Reduce(&my_counter, &final_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_RESULTS);
		if (mpi_rank == 0) {
			printf("%d\n", final_count);
		}
	}

	CP;

	if (print_stat == 1) {
		double commtime = comm_end - comm_start;
		double comptime = comp_end - comp_start;

		vector<double> commtimes(num_processes);
		vector<double> comptimes(num_processes);

		MPI_Gather(&commtime, 1, MPI_DOUBLE, commtimes.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Gather(&comptime, 1, MPI_DOUBLE, comptimes.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		// stat in an easy to import into R format
		if (mpi_rank == 0) {
			printf("comm\t%d\t%d\t%d\t%d\t%d\t%d\t", sparse_mode, num_processes, repl_fact, exponent, N,
					(int) sparse.cells.size());
			for (auto t : commtimes)
				printf("%lf\t", t);
			printf("\n");
			printf("comp\t%d\t%d\t%d\t%d\t%d\t%d\t", sparse_mode, num_processes, repl_fact, exponent, N,
					(int) sparse.cells.size());
			for (auto t : comptimes)
				printf("%lf\t", t);
			printf("\n");
		}
	}

	CP;
	MPI_Type_free(&MPI_SPARSE_CELL);
	CP;
	MPI_Finalize();
	CP;
	return 0;
}
