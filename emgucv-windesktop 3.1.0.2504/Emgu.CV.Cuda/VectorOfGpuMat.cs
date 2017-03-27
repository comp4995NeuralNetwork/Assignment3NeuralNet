﻿//----------------------------------------------------------------------------
//
//  Copyright (C) 2004-2016 by EMGU Corporation. All rights reserved.
//
//  Vector of GpuMat
//
//  This file is automatically generated, do not modify.
//----------------------------------------------------------------------------

using System;
using System.Drawing;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Emgu.CV.Structure;

namespace Emgu.CV.Cuda
{
   /// <summary>
   /// Wrapped class of the C++ standard vector of GpuMat.
   /// </summary>
   public partial class VectorOfGpuMat : Emgu.Util.UnmanagedObject, IInputOutputArray
   {
      private readonly bool _needDispose;
   
      static VectorOfGpuMat()
      {
         CvInvoke.CheckLibraryLoaded();
      }

      /// <summary>
      /// Create an empty standard vector of GpuMat
      /// </summary>
      public VectorOfGpuMat()
         : this(VectorOfGpuMatCreate(), true)
      {
      }
	  
	   internal VectorOfGpuMat(IntPtr ptr, bool needDispose)
      {
         _ptr = ptr;
         _needDispose = needDispose;
      }

      /// <summary>
      /// Create an standard vector of GpuMat of the specific size
      /// </summary>
      /// <param name="size">The size of the vector</param>
      public VectorOfGpuMat(int size)
         : this( VectorOfGpuMatCreateSize(size), true)
      {
      }
	  
	   /// <summary>
      /// Create an standard vector of GpuMat with the initial values
      /// </summary>
      /// <param name="values">The initial values</param>
	   public VectorOfGpuMat(params GpuMat[] values)
	     : this()
	   {
         Push(values);
	   }

      /// <summary>
      /// Get the size of the vector
      /// </summary>
      public int Size
      {
         get
         {
            return VectorOfGpuMatGetSize(_ptr);
         }
      }

      /// <summary>
      /// Clear the vector
      /// </summary>
      public void Clear()
      {
         VectorOfGpuMatClear(_ptr);
      }
	  
	   /// <summary>
      /// Push a value into the standard vector
      /// </summary>
      /// <param name="value">The value to be pushed to the vector</param>
      public void Push(GpuMat value)
      {
         VectorOfGpuMatPush(_ptr, value.Ptr);
      }

      /// <summary>
      /// Push multiple values into the standard vector
      /// </summary>
      /// <param name="values">The values to be pushed to the vector</param>
      public void Push(GpuMat[] values)
      {
         foreach (GpuMat value in values)
            Push(value);
      }

      /// <summary>
      /// Push multiple values from the other vector into this vector
      /// </summary>
      /// <param name="other">The other vector, from which the values will be pushed to the current vector</param>
      public void Push(VectorOfGpuMat other)
      {
         VectorOfGpuMatPushVector(_ptr, other);
      }
      
	   /// <summary>
      /// Get the item in the specific index
      /// </summary>
      /// <param name="index">The index</param>
      /// <returns>The item in the specific index</returns>
      public GpuMat this[int index]
      {
         get
         {
		    IntPtr itemPtr = IntPtr.Zero;
            VectorOfGpuMatGetItemPtr(_ptr, index, ref itemPtr);
            return new GpuMat(itemPtr, false);
         }
      }

      /// <summary>
      /// Release the standard vector
      /// </summary>
      protected override void DisposeObject()
      {
         if (_needDispose && _ptr != IntPtr.Zero)
            VectorOfGpuMatRelease(ref _ptr);
      }

	   /// <summary>
      /// Get the pointer to cv::_InputArray
      /// </summary>
      public InputArray GetInputArray()
      {
        return new InputArray( cvInputArrayFromVectorOfGpuMat(_ptr), this );
      }
	  
      /// <summary>
      /// Get the pointer to cv::_OutputArray
      /// </summary>
      public OutputArray GetOutputArray()
      {
         return new OutputArray( cvOutputArrayFromVectorOfGpuMat(_ptr), this );
      }

	   /// <summary>
      /// Get the pointer to cv::_InputOutputArray
      /// </summary>
      public InputOutputArray GetInputOutputArray()
      {
         return new InputOutputArray( cvInputOutputArrayFromVectorOfGpuMat(_ptr), this );
      }     
      
      /// <summary>
      /// The size of the item in this Vector, counted as size in bytes.
      /// </summary>
      public static int SizeOfItemInBytes
      {
         get { return VectorOfGpuMatSizeOfItemInBytes(); }
      }

#if false
      /// <summary>
      /// Create the standard vector of GpuMat 
      /// </summary>
      public VectorOfGpuMat(MDMatch[][] values)
         : this()
      {
         using (GpuMat v = new GpuMat())
         {
            for (int i = 0; i < values.Length; i++)
            {
               v.Push(values[i]);
               Push(v);
               v.Clear();
            }
         }
      }
	  
	   /// <summary>
      /// Convert the standard vector to arrays of int
      /// </summary>
      /// <returns>Arrays of int</returns>
      public MDMatch[][] ToArrayOfArray()
      {
         int size = Size;
         MDMatch[][] res = new MDMatch[size][];
         for (int i = 0; i < size; i++)
         {
            using (GpuMat v = this[i])
            {
               res[i] = v.ToArray();
            }
         }
         return res;
      }
#endif

      [DllImport(CvInvoke.ExternLibrary, CallingConvention = CvInvoke.CvCallingConvention)]
      internal static extern IntPtr VectorOfGpuMatCreate();

      [DllImport(CvInvoke.ExternLibrary, CallingConvention = CvInvoke.CvCallingConvention)]
      internal static extern IntPtr VectorOfGpuMatCreateSize(int size);

      [DllImport(CvInvoke.ExternLibrary, CallingConvention = CvInvoke.CvCallingConvention)]
      internal static extern void VectorOfGpuMatRelease(ref IntPtr v);

      [DllImport(CvInvoke.ExternLibrary, CallingConvention = CvInvoke.CvCallingConvention)]
      internal static extern int VectorOfGpuMatGetSize(IntPtr v);

      [DllImport(CvInvoke.ExternLibrary, CallingConvention = CvInvoke.CvCallingConvention)]
      internal static extern void VectorOfGpuMatPush(IntPtr v, IntPtr value);

      [DllImport(CvInvoke.ExternLibrary, CallingConvention = CvInvoke.CvCallingConvention)]
      internal static extern void VectorOfGpuMatPushVector(IntPtr ptr, IntPtr otherPtr);
      
      [DllImport(CvInvoke.ExternLibrary, CallingConvention = CvInvoke.CvCallingConvention)]
      internal static extern void VectorOfGpuMatClear(IntPtr v);

      [DllImport(CvInvoke.ExternLibrary, CallingConvention = CvInvoke.CvCallingConvention)]
      internal static extern void VectorOfGpuMatGetItemPtr(IntPtr vec, int index, ref IntPtr element);

      [DllImport(CvInvoke.ExternLibrary, CallingConvention = CvInvoke.CvCallingConvention)]
      internal static extern int VectorOfGpuMatSizeOfItemInBytes();

      [DllImport(CvInvoke.ExternLibrary, CallingConvention = CvInvoke.CvCallingConvention)]
      internal static extern IntPtr cvInputArrayFromVectorOfGpuMat(IntPtr vec);

      [DllImport(CvInvoke.ExternLibrary, CallingConvention = CvInvoke.CvCallingConvention)]
      internal static extern IntPtr cvOutputArrayFromVectorOfGpuMat(IntPtr vec);

      [DllImport(CvInvoke.ExternLibrary, CallingConvention = CvInvoke.CvCallingConvention)]
      internal static extern IntPtr cvInputOutputArrayFromVectorOfGpuMat(IntPtr vec);
   }
}
