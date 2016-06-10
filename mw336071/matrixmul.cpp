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

#include <mpi.h>

#include "densematgen.h"

#define checkpointon false
#define debon false
#define SEQ {this_thread::sleep_for (std::chrono::seconds(mpi_rank)); }
#define CP {if (checkpointon) cout << endl << "CHECKPOINT@" << __LINE__ << endl;}
#define deb(burak) if(debon) {cout<<__LINE__<< " DEB-> "<<#burak<<": "<<burak<<endl;}
#define debv(burak) if(debon) {cout<<__LINE__<< " DEB-> "<<#burak<<": \t"; for(unsigned int zyx=0;zyx<burak.size();zyx++) cout<<burak[zyx]<<" "; cout<<endl;}
#define debt(burak,SIzE) if(debon) {cout<<__LINE__<< " DEB-> "<<#burak<<": \t"; for(unsigned int zyx=0;zyx<SIzE;zyx++) cout<<burak[zyx]<<" "; cout<<endl;}
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

	DenseMatrix() :
			Matrix() {
	}

	// Initialize matrix with 0
	DenseMatrix(int nrow, int ncol, int start_row = 0, int start_col = 0) :
			Matrix() {
		this->cells = new double[nrow * ncol](); // continous block of memory

		this->nrow = nrow;
		this->ncol = ncol;
		this->start_row = start_row;
		this->start_col = start_col;
		this->last_row_excl = start_row + nrow;
		this->last_col_excl = start_col + ncol;
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
		/*for (auto& row : cells) {
		 for (auto & col : row) {
		 col = v;
		 }
		 }*/
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
		// may sort cells in some order
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

	int show_results = 0, use_inner = 0, gen_seed = -1, repl_fact = 1, option = -1, exponent = 1, count_ge = 0,
			sparse_mode = 0, sparse_mode_default = 1, print_stat = 0;
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
				use_inner = 1;
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
	if ((gen_seed == -1) || ((mpi_rank == 0) && sparse.isEmpty())) {
		fprintf(stderr, "error: missing seed or sparse matrix file; exiting\n");
		MPI_Finalize();
		return 3;
	}

	if (sparse_mode_default == 1) {
		if (use_inner == 1) {
			sparse_mode = 3;
		} else {
			sparse_mode = 0;
		}
	}

	if (use_inner == 0) {
		if (num_processes % repl_fact != 0) {
			fprintf(stderr, "error: replication factor should divide number of processes; exiting\n");
			MPI_Finalize();
			return 3;
		}

	}
	CP;

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
	MPI_Request repl_req;

	CP

		// waits for finish following after generating dense matrix
	MPI_Iallgatherv(my_sparse_chunk.data(), my_sparse_chunk.size(), MPI_SPARSE_CELL, repl_sparse_chunk.data(),
			chunks_sizes, chunks_displs, MPI_SPARSE_CELL, MPI_COMM_REPL, &repl_req);

	comm_end = MPI_Wtime();
	CP;

	if ((debon)) {
		printf("Communication %d: %.5f\n", mpi_rank, comm_end - comm_start);
	}
	DenseMatrix my_dense = generateDenseSubmatrix(mpi_rank, num_processes, N, gen_seed);
	DenseMatrix partial_res(my_dense); // initialize with 0

	SparseMatrix my_sparse;
	MPI_Wait(&repl_req, MPI_STATUS_IGNORE);
	my_sparse.cells = std::move(repl_sparse_chunk);

	// COMPUTE RESULT MATRIX
	const int rounds_num = num_processes / repl_fact;
	const int next_proc = (mpi_rank + repl_fact) % num_processes;
	const int prev_proc = (mpi_rank - repl_fact + num_processes) % num_processes;
	MPI_Request reqs[2];
	MPI_Status stats[2];
	int incoming_size;
	sparse_type sparse_buff;

	CP;

	comp_start = MPI_Wtime();
	for (int exp_i = 0; exp_i < exponent; ++exp_i) {
		partial_res.setCells(0);

		for (int r = 0; r < rounds_num; ++r) {
			MPI_Isend(my_sparse.cells.data(), my_sparse.cells.size(), MPI_SPARSE_CELL, next_proc, 0, MPI_COMM_WORLD,
					reqs + 0);

			partialMul(partial_res, my_sparse, my_dense); // does not modify my_sparse

			MPI_Probe(prev_proc, 0, MPI_COMM_WORLD, stats + 0);
			MPI_Get_count(stats + 0, MPI_SPARSE_CELL, &incoming_size);
			sparse_buff.resize(incoming_size);
			MPI_Irecv(sparse_buff.data(), incoming_size, MPI_SPARSE_CELL, prev_proc, 0, MPI_COMM_WORLD, reqs + 1);

			MPI_Waitall(2, reqs, stats);

			for (int st = 0; st < 2; st++) {
				if (stats[st].MPI_ERROR != MPI_SUCCESS) {
					throw runtime_error("Error in non-blocking send/receive: " + to_string(st));
				}
			}

			swap(my_sparse.cells, sparse_buff); // O(1)
			sparse_buff.clear();

		}

		swap(my_dense.cells, partial_res.cells); // O(1)
	}
	comp_end = MPI_Wtime();

	if ( debon) {
		printf("Computations %d: %.5f\n", mpi_rank, comp_end - comp_start);
	}
	CP;
	if (show_results) {
		// gather results to root
		int s = (mpi_rank == 0) ? N : 0;
		int rcounts[s];
		int displs[s];
		double* result = NULL;
		CP;

		if (mpi_rank == 0) {
			result = new double[s * s]; // on heap
			//double result[N*N]; // on stack;
			for (int i = 0; i < num_processes; ++i) {
				auto info = blocked1DSubMatrixInfo(i, num_processes, N);
				rcounts[i] = info.ncol * info.nrow; /* note change from previous example */
				displs[i] = (i == 0) ? 0 : displs[i - 1] + rcounts[i - 1];
			}
		}

		MPI_Gatherv(my_dense.cells, my_dense.nrow * my_dense.ncol, MPI_DOUBLE, result, rcounts, displs, MPI_DOUBLE, 0,
		MPI_COMM_WORLD);

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

	if (count_ge) {
		int my_counter = 0, final_count;
		for (int i = 0; i < my_dense.nrow * my_dense.ncol; i++) {
			my_counter += int(my_dense.cells[i] >= ge_element);
		}
		MPI_Reduce(&my_counter, &final_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
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
