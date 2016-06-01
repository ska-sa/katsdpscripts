#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* INITIALIZATION OF FUNCTIONS ... */
void send_string(char *string); /* functions used to write "filterbank" PRESTO data files */
void send_float(char *name, float floating_point); /* ditto */
void send_double(char *name, double double_precision); /* ditto */
void send_long(char *name, long integer); /* ditto */
void send_int(char *name, int integer); /* ditto */
void send_coords(double raj, double dej, double az, double za); /* ditto */
double gregjd(int jy, int jm, int jd, double rh, double rm, double rs); /* convert date to Julian Date */

/* ... AND GLOBAL VARIABLES */
FILE *FILE_OUT; /* pointer to output file */

/* MAIN BODY OF THE PROGRAM */
int main(int argc, char *argv[]) {

   int telescope_id, machine_id, data_type, barycentre;
   int pulsarcentre, nbits, nsamples, nchans, nifs, argument;
   int RawDataFile, RefDM, Period;
   double az_start, za_start, src_raj, src_dej, tstart, tsamp, fch1, fo, refdm, period;
   double RAh, RAm, RAs, DecD, DecM, DecS;
   char source_name[64], raw_data_file[128];
   char RAtxt[16], DecTxt[16], out_filename[256];
   time_t GMT_Time; /* declare a variable 'GMT_Time' which is data type of 'time_t' to store calendar time */
   struct tm *Tm; /* structure tm contains a calendar date and time broken down into its components, global declaration of Tm as variable of tm type */

   /* PRESET FLAGS & DEFAULT VALUES */
   telescope_id = 64;
   machine_id = 64;
   data_type = 1;
   barycentre = 0;
   pulsarcentre = 0;
   nbits = 32;
   nsamples = 8192;
   nchans = 512;
   nifs = 1;
   strcpy(source_name, "J0000+0000");
   az_start = 0.0;
   za_start = 0.0;
   src_raj = 0.0;
   src_dej = 0.0;
   GMT_Time=time(NULL); /* get current calendar time and assign it to 'GMT_Time' */
   Tm=gmtime(&GMT_Time); /* use function gmtime to transform date and time to broken-down time */
   tstart = gregjd(Tm->tm_year+1900, Tm->tm_mon+1, Tm->tm_mday, (double) Tm->tm_hour, (double) Tm->tm_min, Tm->tm_sec) - 2400000.5; /* convert the date to JD and subtract 2400000.5 to get MJD */
   tsamp = 0.000089200;
   fch1 = 1922;
   fo = -0.390625;
   refdm = 0.0;
   period = 0.0;
   RawDataFile = 0;
   RefDM = 0;

   /* AVAILABLE HELP */
   if (argc < 2 || (strcmp(argv[1], "-h") == 0)) { /* in case there is a -h switch */
      printf("\nProgram for preparing filterbank header. If values are not given\n");
      printf("then the program will use default values and produce a header. Not\n");
      printf("all header variables are compulsory but most of them are desired.\n");
      printf("\nUsage: mockHeader [parameters] <header_filename>\n");
      printf("\nAvailable parameters are:\n");
      printf("-tel      Telescope identifier (FR606: 12, DE601: 13, UK608: 14, etc.; default: %d)\n", telescope_id);
      printf("-mach     Machine identifier (Fake: 0, ARTEMIS: 10, DSP-Z: 12, etc.; default: %d)\n", machine_id);
      printf("-type     Data type identifier (Raw data: 0, Filterbank: 1, etc.; default: %d)\n", data_type);
      printf("-raw      The name of the original data file (default: unset)\n");
      printf("-source   Name of the source to be written in header file (default: %s)\n", source_name);
      printf("-bary     Equals 1 if data is barycentric or 0 if otherwise (default: %d)\n", barycentre);
      printf("-puls     Equals 1 if data is pulsarcentric or 0 if otherwise (default: %d)\n", pulsarcentre);
      printf("-az       Telescope azimuth at start of scan in degrees (default: %f)\n", az_start);
      printf("-za       Telescope zenith angle at start of scan in degrees (default: %f)\n", za_start);
      printf("-ra       Right Ascention (J2000) of the source in hh:mm:ss.s (default: %f)\n", src_raj);
      printf("-dec      Declination (J2000) of the source in dd:mm:ss.s (default: %f)\n", src_dej);
      printf("-tstart   Time stamp (MJD) of first sample (default [current UTC time]: %.5f)\n", tstart);
      printf("-tsamp    Time interval between samples in sec (default: %.8f)\n", tsamp);
      printf("-nbits    Number of bits per time sample (default: %d)\n", nbits);
      printf("-fch1     Centre frequency in MHz of first filterbank channel (default: %f)\n", fch1);
      printf("-fo       Filterbank channel bandwidth in MHz (default: %f)\n", fo);
      printf("-nchans   Number of filterbank channels (default: %d)\n", nchans);
      printf("-nifs     Number of seperate IF channels (default: %d)\n", nifs);
      printf("-refdm    Reference dispersion measure in pc/ccm (default: unset)\n");
      printf("-period   Folding period in sec (default: unset)\n");
      printf("-h        Display this useful help page\n\n");
      return -1;
   }

   /* READING COMMAND LINE ARGUMENTS */
   if (argc > 2) {
      for (argument = 1; argument < argc - 1; argument++) {
         if (strcmp(argv[argument], "-tel") == 0) {
            telescope_id = atoi(argv[argument+1]);
            argument++;
         } else if (strcmp(argv[argument], "-mach") == 0)  {
            machine_id = atoi(argv[argument+1]);
            argument++;
         } else if (strcmp(argv[argument], "-type") == 0)  {
            data_type = atoi(argv[argument+1]);
            argument++;
         } else if (strcmp(argv[argument], "-raw") == 0) {
            strcpy(raw_data_file, argv[argument+1]);
            argument++;
            RawDataFile = 1;
            printf("%s\n", raw_data_file);
         } else if (strcmp(argv[argument], "-source") == 0) {
            strcpy(source_name, argv[argument+1]);
            argument++;
         } else if (strcmp(argv[argument], "-bary") == 0)  {
            barycentre = atoi(argv[argument+1]);
            argument++;
         } else if (strcmp(argv[argument], "-puls") == 0)  {
            pulsarcentre = atoi(argv[argument+1]);
            argument++;
         } else if (strcmp(argv[argument], "-az") == 0) {
            az_start = atof(argv[argument+1]);
            argument++;
         } else if (strcmp(argv[argument], "-za") == 0) {
            za_start = atof(argv[argument+1]);
            argument++;
         } else if (strcmp(argv[argument], "-ra") == 0) {
            strcpy(RAtxt, argv[argument+1]);
            sscanf(RAtxt, "%lf:%lf:%lf", &RAh, &RAm, &RAs);
            src_raj = (RAh * 10000.0) + (RAm * 100.0) + RAs;
            if (src_raj < 0.0 || src_raj > 240000.0) {
               fprintf(stderr, "RA of %f degrees does not make sense.\n", src_raj);
               return -1;
            }
            argument++;
         } else if (strcmp(argv[argument], "-dec") == 0) {
            strcpy(DecTxt, argv[argument+1]);
            sscanf(DecTxt, "%lf:%lf:%lf", &DecD, &DecM, &DecS);
            if (strncmp (DecTxt,"-",1) == 0) {
               src_dej = (DecD * 10000.0) + (DecM * -100.0) + (-1.0 * DecS);
            } else {
               src_dej = (DecD * -10000.0) + (DecM * -100.0) + ( DecS * -1.0);
            }
            if (src_dej < -900000.0 || src_dej >= 900000.0) {
               fprintf(stderr, "Dec value does not make sense.\n");
               return -1;
            }
            argument++;
         } else if (strcmp(argv[argument], "-tstart") == 0) {
            tstart = atof(argv[argument+1]);
            argument++;
         } else if (strcmp(argv[argument], "-tsamp") == 0) {
            tsamp = atof(argv[argument+1]);
            argument++;
         } else if (strcmp(argv[argument], "-nbits") == 0) {
            nbits = atoi(argv[argument+1]);
            argument++;
         } else if (strcmp(argv[argument], "-fch1") == 0) {
            fch1 = atof(argv[argument+1]);
            argument++;
         } else if (strcmp(argv[argument], "-fo") == 0) {
            fo = atof(argv[argument+1]);
            argument++;
         } else if (strcmp(argv[argument], "-nchans") == 0) {
            nchans = atoi(argv[argument+1]);
            argument++;
         } else if (strcmp(argv[argument], "-nifs") == 0) {
            nifs = atoi(argv[argument+1]);
            argument++;
         } else if (strcmp(argv[argument], "-refdm") == 0) {
            refdm = atof(argv[argument+1]);
            RefDM = 1;
            argument++;
         } else if (strcmp(argv[argument], "-period") == 0) {
            period = atof(argv[argument+1]);
            Period = 1;
            argument++;
         } else {
            fprintf(stderr, "Unknown option: %s\n", argv[argument]);
            return -1;
         }
      }
   }
   strcpy(out_filename, argv[argc-1]);
   if ((FILE_OUT = fopen(out_filename, "w")) == NULL) {
      fprintf(stderr, "%s> Unable to open file %s\n", argv[0], out_filename);
      return -1;
   }
   send_string("HEADER_START");
   if (RawDataFile == 1) {
      send_string("rawdatafile"); /* setting... */
      send_string(raw_data_file); /* ... name of the original data file */
   }
   send_string("source_name"); /* setting... */
   send_string(source_name); /* ... name of the source being observed by the telescope */
   send_int("machine_id", machine_id); /* ID of datataking machine */
   send_int("telescope_id", telescope_id); /* ID of telescope */
   send_coords(src_raj, src_dej, az_start, za_start); /* RA (J2000), Dec (J2000), Az, ZA */
   send_int("data_type", data_type); /* data type */
   send_double("fch1", fch1); /* centre frequency (MHz) of first filterbank channel */
   send_double("foff", fo); /* filterbank channel bandwidth (MHz) */
   send_int("nchans", nchans); /* number of filterbank channels */
   send_int("nbits", nbits); /* number of bits per time sample */
   send_double("tstart", tstart); /* time stamp (MJD) of first sample */
   send_double("tsamp", tsamp); /* time interval between samples (s) */
   send_int("nifs", nifs); /* number of seperate IF channels */
   if (Period == 1) {
      send_double("period", period); /* time interval between samples (s) */
   }
   if (RefDM == 1) {
      send_double("refdm", refdm); /* time interval between samples (s) */
   }
   send_string("HEADER_END");

   fclose(FILE_OUT);
   printf("%s> Created header file %s\n", argv[0], out_filename);
   return 0;
}

void send_string(char *string) {
   int len;
   len = strlen(string);
   fwrite(&len, sizeof(int), 1, FILE_OUT);
   fwrite(string, sizeof(char), len, FILE_OUT);
}

void send_float(char *name, float floating_point) {
   send_string(name);
   fwrite(&floating_point, sizeof(float), 1, FILE_OUT);
}

void send_double (char *name, double double_precision) {
   send_string(name);
   fwrite(&double_precision, sizeof(double), 1, FILE_OUT);
}

void send_long(char *name, long integer) {
   send_string(name);
   fwrite(&integer, sizeof(long), 1, FILE_OUT);
}

void send_int(char *name, int integer) {
   send_string(name);
   fwrite(&integer, sizeof(int), 1, FILE_OUT);
}

void send_coords(double raj, double dej, double az, double za) {
  if ((raj != 0.0) || (raj != -1.0)) send_double("src_raj", raj);
  if ((dej != 0.0) || (dej != -1.0)) send_double("src_dej", dej);
  if ((az != 0.0)  || (az != -1.0))  send_double("az_start", az);
  if ((za != 0.0)  || (za != -1.0))  send_double("za_start", za);
}

double gregjd(int jy, int jm, int jd, double rh, double rm, double rs) {
   double rj,ra,rb,rg,tjd,y,m;
   rj=jd+rh/24.0+rm/1440.0+rs/86400.0;
   if(jm<=2) {
      y=jy-1;
      m=jm+12;
   } else {
      y=jy;
      m=jm;
   }
   rg=y+m/100+rj/10000;
   ra=0.0;
   rb=0.0;
   if (rg>=1582.1015) {
      ra=floor(y/100.0);
      rb=2-ra+floor(ra/4.0);
   }
   tjd=floor(365.25*y) + floor(30.6001*(m+1)) + rj +1720994.5 + rb;
   return tjd;
}
