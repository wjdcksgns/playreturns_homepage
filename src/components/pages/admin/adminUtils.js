import * as XLSX from 'xlsx';

export const excelToCsvFile = (file, forcedFileName) => {
    return new Promise((resolve) => {
        const reader = new FileReader();

        reader.onload = (e) => {
            const data = new Uint8Array(e.target.result);
            const workbook = XLSX.read(data, { type: 'array' });

            const sheetName = workbook.SheetNames[0];
            const worksheet = workbook.Sheets[sheetName];

            const csv = XLSX.utils.sheet_to_csv(worksheet);

            const csvFile = new File(
                [csv],
                forcedFileName, // ✅ 파일명 강제
                { type: 'text/csv;charset=utf-8;' }
            );

            resolve(csvFile);
        };

        reader.readAsArrayBuffer(file);
    });
};
