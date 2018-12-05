
 #include <stdio.h>
 main()
 {
 FILE *fp1, *fp2;
 short c;
 int i, j, header, x, y;
 char file1[20], file2[20];

 /* ask questions */
 printf("enter binary file name : ");
 gets(file1);
 printf("enter integer file name : ");
 gets(file2);
 printf("enter header size of the binary file : ");
 scanf("%d",&header);
 printf("enter the size of the image : x * y \n");
 printf("x = "); scanf("%d",&x);
 printf("y = "); scanf("%d",&y);

 if((fp1 = fopen(file1, "rb")) == NULL)
   {printf("cannot open %s\n", file1);
    exit(1);
   }
 if((fp2 = fopen(file2, "w+b")) == NULL)
   {printf("cannot open %s\n", file2);
    exit(1);
   }
 for (i = 1; i <= header; i++)  /* skip the header */
   c = getc(fp1);

 fprintf(fp2,"%4d",x);  fprintf(fp2,"%4d",y);
 for (i = 1; i <= y; i++)
   for (j = 1; j <= x; j++)
     {
      c = getc(fp1);
      fprintf(fp2,"%4d",c);
     }
 fclose(fp1);
 fclose(fp2);
}