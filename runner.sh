#SBATCH --mail-user=jhgearon@iu.edu
#SBATCH --nodes=1
#SBATCH -A r00268
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-9:59:00
#SBATCH --mem=32gb
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --job-name=army
#SBATCH --output=army.out
#SBATCH --error=army.err


module load python
module load gdal

# Set the working directory
pip install -r requirements.txt

python army_levees/nld_api.py